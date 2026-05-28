import json
import hashlib
from pathlib import Path
import re
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "kietnt0603/nrk-legal-large"
MAX_TOKENS = 400

model = SentenceTransformer(MODEL_NAME, device='cpu')
model.max_seq_length = 512
tokenizer = model.tokenizer

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def clean_italic_markers(text: str) -> str:
    return re.sub(r'\*([^*]+)\*', r'\1', text)

def drop_amended_lines(text: str) -> str:
    lines = text.split('\n')
    valid_lines = []

    for line in lines:
        if "[[AMENDMENT:" in line:
            anchor_match = re.search(r'\[\[ANCHOR:[^\]]+\]\]', line)
            header_match = re.search(r'\*(Phần|Chương|Mục|Điều)\s+[^*]+\*', line)

            preserved_elements = []
            if anchor_match:
                preserved_elements.append(anchor_match.group(0))
            if header_match:
                preserved_elements.append(header_match.group(0))

            if preserved_elements:
                valid_lines.append(" ".join(preserved_elements))
        else:
            valid_lines.append(line)

    return '\n'.join(valid_lines)

def extract_and_clean_markers(text: str):
    anchor_ids = re.findall(r'\[\[ANCHOR:([^\]]+)\]\]', text)

    clean_text = re.sub(r'\[\[(ANCHOR):[^\]]+\]\]\s*', '', text).strip()
    clean_text = re.sub(r'\[\[(AMENDMENT):[^\]]+\]\]\s*', '', clean_text).strip()

    clean_text = re.sub(r'[ \t\xa0]+', ' ', clean_text).strip()
    clean_text = re.sub(r'\n{2,}', '\n', clean_text)

    return clean_text, anchor_ids

def hierarchical_chunk(json_data: dict) -> list:
    chunks = []
    doc_id = hashlib.md5(json_data["source_url"].encode()).hexdigest()[:16]
    clean_main_title = clean_italic_markers(json_data["title"])
    base_meta = {
        "doc_id": doc_id, "source_url": json_data["source_url"], "title": clean_main_title,
        **json_data.get("metadata", {}),
    }

    meta_dict = json_data.get("metadata", {})

    meta_text = (
        f"Thông tin chung (Metadata) của văn bản pháp luật:\n"
        f"- Tên văn bản: {clean_main_title}\n"
        f"- Loại văn bản: {meta_dict.get('Loại văn bản', 'Không có thông tin')}\n"
        f"- Nơi ban hành: {meta_dict.get('Nơi ban hành', 'Không có thông tin')}\n"
        f"- Người ký: {meta_dict.get('Người ký', 'Không có thông tin')}\n"
        f"- Ngày ban hành: {meta_dict.get('Ngày ban hành', 'Không có thông tin')}\n"
        f"- Ngày hiệu lực: {meta_dict.get('Ngày hiệu lực', 'Không có thông tin')}\n"
        f"- Tình trạng: {meta_dict.get('Tình trạng', 'Không có thông tin')}\n"
        f"- Số công báo: {meta_dict.get('Số công báo', 'Không có thông tin')}\n"
        f"- Ngày công báo: {meta_dict.get('Ngày công báo', 'Không có thông tin')}\n"
    )

    chunks.append({
        **base_meta,
        "chunk_type": "Document_Meta",
        "parent_chunk_id": doc_id,
        "content": meta_text,
        "content_embed": meta_text,
        "chunk_id": f"meta_{doc_id}"
    })

    summary = json_data.get("summary_content", "").strip()
    if summary:
        summary_header = f"Tóm tắt nội dung văn bản: {clean_main_title}\n\n"
        full_summary_text = summary_header + summary

        if count_tokens(full_summary_text) <= MAX_TOKENS:
            chunks.append({
                **base_meta,
                "chunk_type": "Summary",
                "parent_chunk_id": doc_id,
                "content": summary,
                "content_embed": full_summary_text,
                "chunk_id": f"summary_{doc_id}"
            })
        else:
            summary_parts = re.split(r'\n\n+', summary)
            buf_raw, p_idx = "", 1

            for p_raw in summary_parts:
                test_text = summary_header + (f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw)

                if count_tokens(test_text) > MAX_TOKENS and buf_raw:
                    chunks.append({
                        **base_meta,
                        "chunk_type": "Summary_Split",
                        "parent_chunk_id": doc_id,
                        "content": buf_raw,
                        "content_embed": summary_header + buf_raw,
                        "chunk_id": f"summary_{doc_id}_part{p_idx}"
                    })
                    buf_raw, p_idx = p_raw, p_idx + 1
                else:
                    buf_raw = f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw

            if buf_raw:
                chunks.append({
                    **base_meta,
                    "chunk_type": "Summary_Split",
                    "parent_chunk_id": doc_id,
                    "content": buf_raw,
                    "content_embed": summary_header + buf_raw,
                    "chunk_id": f"summary_{doc_id}_part{p_idx}"
                })

    main_content = json_data.get("main_content", "")
    main_content = drop_amended_lines(main_content)

    anchors_iter = list(re.finditer(r'\[\[ANCHOR:[^\]]+\]\]', main_content))
    segments = []
    start_idx = 0
    for match in anchors_iter:
        if match.start() > start_idx:
            segments.append(main_content[start_idx:match.start()])
        start_idx = match.start()
    if start_idx < len(main_content):
        segments.append(main_content[start_idx:])

    current_phan_id = doc_id
    current_chuong_id = doc_id
    current_muc_id = doc_id

    parent_text = f"Văn bản: {clean_main_title}\n\nThuộc: "

    dieu_buffer, dieu_tokens = [], 0
    khoan_buffer, khoan_tokens = [], 0
    diem_buffer, diem_tokens = [], 0

    def flush_dieu():
        nonlocal dieu_buffer, dieu_tokens, chunks
        if not dieu_buffer: return
        combined_text = "\n\n".join([d["content"] for d in dieu_buffer])
        chunk_id = f"{dieu_buffer[0]['chunk_id']}_gop" if len(dieu_buffer) > 1 else dieu_buffer[0]["chunk_id"]

        combined_anchors = []
        for d in dieu_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
        unique_anchors = list(set(combined_anchors))

        chunks.append({
            **base_meta,
            "chunk_type": "Điều",
            "parent_chunk_id": dieu_buffer[0]["parent_id"],
            "content": combined_text,
            "content_embed": f"{parent_text}{combined_text}",
            "chunk_id": chunk_id,
            "anchors": unique_anchors
        })
        dieu_buffer, dieu_tokens = [], 0

    def flush_khoan(parent_id, cau_dan_clean):
        nonlocal khoan_buffer, khoan_tokens, chunks
        if not khoan_buffer: return
        combined_text = "\n\n".join([k["content"] for k in khoan_buffer])
        chunk_id = f"{khoan_buffer[0]['chunk_id']}_gop" if len(khoan_buffer) > 1 else khoan_buffer[0]["chunk_id"]

        combined_anchors = []
        for d in khoan_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
        unique_anchors = list(set(combined_anchors))

        chunks.append({
            **base_meta,
            "chunk_type": "Khoản",
            "parent_chunk_id": parent_id,
            "content": combined_text,
            "content_embed": f"{parent_text}{cau_dan_clean}\n\n{combined_text}",
            "chunk_id": chunk_id,
            "anchors": unique_anchors
        })
        khoan_buffer, khoan_tokens = [], 0

    def flush_diem(parent_id, cau_dan_clean, k_dan_clean):
        nonlocal diem_buffer, diem_tokens, chunks
        if not diem_buffer: return
        combined_text = "\n".join([d["content"] for d in diem_buffer])
        chunk_id = f"{diem_buffer[0]['chunk_id']}_gop" if len(diem_buffer) > 1 else diem_buffer[0]["chunk_id"]

        combined_anchors = []
        for d in diem_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
        unique_anchors = list(set(combined_anchors))

        chunks.append({
            **base_meta,
            "chunk_type": "Điểm",
            "parent_chunk_id": parent_id,
            "content": combined_text,
            "content_embed": f"{parent_text}{cau_dan_clean}\n\n{k_dan_clean}\n\n{combined_text}",
            "chunk_id": chunk_id,
            "anchors": unique_anchors
        })
        diem_buffer, diem_tokens = [], 0

    for i, seg_raw in enumerate(segments):
        raw_cleaned_italic = clean_italic_markers(seg_raw.strip())

        seg_clean, anchors = extract_and_clean_markers(raw_cleaned_italic)
        if not seg_clean or count_tokens(seg_clean) < 10:
            continue

        seg_with_anchors_only = re.sub(r'\[\[ANCHOR:[^\]]+\]\]\s*', '', raw_cleaned_italic).strip()
        seg_id = f"{doc_id}_{anchors[0]}" if anchors else f"{doc_id}_seg_{i}"
        text_for_checking = re.sub(r'\[\[.*?\]\]\s*', '', seg_clean).strip()

        is_phu_luc = any("_pl_" in anchor for anchor in anchors) if anchors else False

        is_phan = re.match(r'^\*(Phần)\s+[^\*]*\*', text_for_checking)
        is_chuong = re.match(r'^\*(Chương)\s+[^\*]*\*', text_for_checking)
        is_muc = re.match(r'^\*(Mục)\s+[^\*]*\*', text_for_checking)

        if is_phan or is_chuong or is_muc:
            flush_dieu()
            if is_phan:
                current_phan_id = current_chuong_id = current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Phần",
                               "parent_chunk_id": doc_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors})
            elif is_chuong:
                current_chuong_id = current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Chương",
                               "parent_chunk_id": current_phan_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors})
            elif is_muc:
                current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Mục",
                               "parent_chunk_id": current_chuong_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors})
            continue

        if re.match(r'^\*Điều\s+\d+\.[^\n]*', text_for_checking):
            dieu_id = seg_id
            tokens = count_tokens(seg_clean)

            if tokens <= MAX_TOKENS:
                if dieu_tokens + tokens > MAX_TOKENS:
                    flush_dieu()
                dieu_buffer.append(
                    {"content": seg_clean,
                     "chunk_id": dieu_id,
                     "parent_id": current_muc_id,
                     "anchors": anchors})
                dieu_tokens += tokens
            else:
                flush_dieu()
                khoan_parts_raw = re.split(r'\n(?=(?:Khoản\s+\d+|\d+\.))', seg_with_anchors_only)

                if len(khoan_parts_raw) > 1:
                    cau_dan_clean, _ = extract_and_clean_markers(khoan_parts_raw[0])
                    chunks.append({
                        **base_meta,
                        "chunk_type": "Điều",
                        "parent_chunk_id": current_muc_id,
                        "content": cau_dan_clean,
                        "chunk_id": dieu_id,
                        "anchors": anchors
                    })

                    for j, kp_raw in enumerate(khoan_parts_raw[1:], start=1):
                        k_clean, _ = extract_and_clean_markers(kp_raw)
                        khoan_id = f"{seg_id}_k{j}"
                        k_tokens = count_tokens(k_clean)

                        if k_tokens <= MAX_TOKENS:
                            if khoan_tokens + k_tokens > MAX_TOKENS:
                                flush_khoan(dieu_id, cau_dan_clean)
                            khoan_buffer.append({"content": k_clean, "chunk_id": khoan_id, "anchors": anchors})
                            khoan_tokens += k_tokens
                        else:
                            flush_khoan(dieu_id, cau_dan_clean)
                            diem_parts_raw = re.split(r'\n(?=[a-zđ]\))', kp_raw)
                            k_dan_clean, _ = extract_and_clean_markers(diem_parts_raw[0]) if len(diem_parts_raw) > 1 else ("", [])
                            chunks.append({**base_meta, "chunk_type": "Khoản",
                                           "parent_chunk_id": dieu_id,
                                           "content": k_dan_clean, "chunk_id": khoan_id, "anchors": anchors})

                            for k, dp_raw in enumerate(diem_parts_raw[1:] if len(diem_parts_raw) > 1 else diem_parts_raw):
                                d_clean, _ = extract_and_clean_markers(dp_raw)
                                diem_id = f"{seg_id}_k{j}_d{k}"
                                d_tokens = count_tokens(d_clean.strip())

                                if d_tokens <= MAX_TOKENS:
                                    if diem_tokens + d_tokens > MAX_TOKENS:
                                        flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                                    diem_buffer.append({
                                        "content": d_clean.strip(),
                                        "chunk_id": diem_id,
                                        "anchors": anchors
                                    })
                                    diem_tokens += d_tokens
                                else:
                                    flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                                    chunks.append({
                                        **base_meta,
                                        "chunk_type": "Điểm",
                                        "parent_chunk_id": khoan_id,
                                        "content": d_clean.strip(),
                                        "content_embed": f"{parent_text}{cau_dan_clean}\n\n{k_dan_clean}\n\n{d_clean}",
                                        "chunk_id": diem_id,
                                        "anchors": anchors
                                    })

                            flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                    flush_khoan(dieu_id, cau_dan_clean)
                else:
                    flush_dieu()
                    para_parts_raw = re.split(r'\n\n+', seg_with_anchors_only)
                    buf_raw, p_idx = "", 1
                    for p_raw in para_parts_raw:
                        # FALLBACK CẮT BẢNG BIỂU DÀI TRONG ĐIỀU
                        if count_tokens(p_raw) > MAX_TOKENS:
                            if buf_raw:
                                b_clean, _ = extract_and_clean_markers(buf_raw)
                                chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                               "parent_chunk_id": current_muc_id,
                                               "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                               "anchors": anchors})
                                buf_raw = ""
                                p_idx += 1

                            lines = p_raw.split('\n')
                            temp_buf = ""
                            for line in lines:
                                if count_tokens(temp_buf + "\n" + line) > MAX_TOKENS and temp_buf:
                                    b_clean, _ = extract_and_clean_markers(temp_buf)
                                    chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                                   "parent_chunk_id": current_muc_id,
                                                   "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                                   "anchors": anchors})
                                    temp_buf = line
                                    p_idx += 1
                                else:
                                    temp_buf = f"{temp_buf}\n{line}".strip() if temp_buf else line
                            buf_raw = temp_buf
                            continue

                        test_clean, _ = extract_and_clean_markers(buf_raw + "\n\n" + p_raw)
                        if count_tokens(test_clean) > MAX_TOKENS and buf_raw:
                            b_clean, _ = extract_and_clean_markers(buf_raw)
                            chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                           "parent_chunk_id": current_muc_id,
                                           "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                           "anchors": anchors})
                            buf_raw, p_idx = p_raw, p_idx + 1
                        else:
                            buf_raw = f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw

                    if buf_raw:
                        b_clean, _ = extract_and_clean_markers(buf_raw)
                        chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                       "parent_chunk_id": current_muc_id,
                                       "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                       "anchors": anchors})
        else:
            para_parts_raw = re.split(r'\n\n+', seg_with_anchors_only)
            buf_raw, p_idx = "", 1
            for p_raw in para_parts_raw:
                if count_tokens(p_raw) > MAX_TOKENS:
                    if buf_raw:
                        b_clean, _ = extract_and_clean_markers(buf_raw)
                        chunk_item = {**base_meta, "chunk_type": "Preamble", "parent_chunk_id": current_muc_id, "content": b_clean, "chunk_id": f"{seg_id}_p{p_idx}", "anchors": anchors}
                        if not is_phu_luc and anchors: chunk_item["content_embed"] = f"{parent_text}{b_clean}"
                        chunks.append(chunk_item)

                        buf_raw = ""
                        p_idx += 1

                    lines = p_raw.split('\n')
                    temp_buf = ""
                    for line in lines:
                        if count_tokens(temp_buf + "\n" + line) > MAX_TOKENS and temp_buf:
                            b_clean, _ = extract_and_clean_markers(temp_buf)
                            chunk_item = {**base_meta, "chunk_type": "Preamble", "parent_chunk_id": current_muc_id, "content": b_clean, "chunk_id": f"{seg_id}_p{p_idx}", "anchors": anchors}
                            if not is_phu_luc and anchors: chunk_item["content_embed"] = f"{parent_text}{b_clean}"
                            chunks.append(chunk_item)

                            temp_buf = line
                            p_idx += 1
                        else:
                            temp_buf = f"{temp_buf}\n{line}".strip() if temp_buf else line
                    buf_raw = temp_buf
                    continue

                test_clean, _ = extract_and_clean_markers(buf_raw + "\n\n" + p_raw)
                if count_tokens(test_clean) > MAX_TOKENS and buf_raw:
                    b_clean, _ = extract_and_clean_markers(buf_raw)
                    chunk_item = {**base_meta, "chunk_type": "Preamble", "parent_chunk_id": current_muc_id, "content": b_clean, "chunk_id": f"{seg_id}_p{p_idx}", "anchors": anchors}
                    if not is_phu_luc and anchors: chunk_item["content_embed"] = f"{parent_text}{b_clean}"
                    chunks.append(chunk_item)

                    buf_raw, p_idx = p_raw, p_idx + 1
                else:
                    buf_raw = f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw

            if buf_raw:
                b_clean, _ = extract_and_clean_markers(buf_raw)
                chunk_item = {**base_meta, "chunk_type": "Preamble", "parent_chunk_id": current_muc_id, "content": b_clean, "chunk_id": f"{seg_id}_p{p_idx}", "anchors": anchors}
                if not is_phu_luc and anchors: chunk_item["content_embed"] = f"{parent_text}{b_clean}"
                chunks.append(chunk_item)

    flush_dieu()
    return chunks

def chunk_and_save_file(json_file: Path, output_dir: str) -> str:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_list = hierarchical_chunk(data)
    valid_chunks = [c for c in chunk_list]

    if valid_chunks:
        doc_id = valid_chunks[0]["doc_id"]
        output_filename = f"{doc_id}_chunks.json"
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(valid_chunks, f, ensure_ascii=False, indent=4)

        return output_filepath
    return None

if __name__ == "__main__":
    OUTPUT_DIR = os.getenv("DATA_CHUNKED_DIR")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Bắt đầu cắt chunk và lưu vào thư mục: {OUTPUT_DIR}")

    for root, dirs, files in os.walk("data_json_new"):
        for file in files:
            if file.endswith(".json"):
                json_file = Path(os.path.join(root, file))
                saved_path = chunk_and_save_file(json_file, OUTPUT_DIR)
                if saved_path:
                    print(f"  [+] Đã lưu: {saved_path}")

    for root, dirs, files in os.walk("data_json_next"):
        for file in files:
            if file.endswith(".json"):
                json_file = Path(os.path.join(root, file))
                saved_path = chunk_and_save_file(json_file, OUTPUT_DIR)
                if saved_path:
                    print(f"  [+] Đã lưu: {saved_path}")