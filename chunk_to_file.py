import json
import hashlib
from pathlib import Path
import re
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
MAX_TOKENS = 768
MAX_TOKENS_AMEND = 1024

model = SentenceTransformer(MODEL_NAME, device='cpu')
model.max_seq_length = 1024
tokenizer = model.tokenizer
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def clean_italic_markers(text: str) -> str:
    return re.sub(r'\*([^*]+)\*', r'\1', text)

def extract_and_clean_markers(text: str):
    anchor_ids = re.findall(r'\[\[ANCHOR:([^\]]+)\]\]', text)
    amend_ids = re.findall(r'\[\[AMENDMENT:([^\]]+)\]\]', text)
    clean_text = re.sub(r'\[\[(ANCHOR):[^\]]+\]\]\s*', '', text).strip()
    clean_text = re.sub(r'[ \t\xa0]+', ' ', clean_text).strip()
    clean_text = re.sub(r'\n{2,}', '\n', clean_text)
    return clean_text, anchor_ids, amend_ids

def clean_embed_content(text: str) -> str:
    return re.sub(r'\[\[(AMENDMENT):[^\]]+\]\]\s*', '', text).strip()

def process_amendments(amendments_list: list, doc_id: str) -> list:
    processed = []
    for idx, amend in enumerate(amendments_list):
        bm_title = amend.get("bm_title", "").strip()
        amended = amend.get("amended_content", "").strip()
        tip_id = amend.get("tip_id", f"unknown_{idx}")

        clean_amended = re.sub(r'^.*?(như sau:|như sau)\s*(?:[“"”])?\s*', '', amended, flags=re.IGNORECASE).strip()
        clean_amended = re.sub(r'^[“"”]', '', clean_amended).strip()

        context_header = f"**SỬA ĐỔI, BỔ SUNG**\n{bm_title}"
        full_text = f"{context_header}\n\n**Nội dung mới:**\n{clean_amended}"

        if count_tokens(full_text) <= MAX_TOKENS_AMEND:
            processed.append({
                "chunk_type": "Amendment",
                "chunk_id": f"amend_{doc_id}_{tip_id}",
                "tip_id": tip_id,
                "content": full_text.strip(),
                "bm_title": bm_title,
                "amended_link": amend.get("amended_link", "")
            })
        else:
            amend_parts = re.split(r'\n(?=(?:Chương|Điều|Mục|Khoản)\s)', clean_amended)
            buffer_text = ""
            sub_idx = 1

            for part in amend_parts:
                part = part.strip()
                if not part: continue

                test_text = f"{context_header}\n\n**Nội dung mới{' (tiếp theo)' if sub_idx > 1 else ''}:**\n{buffer_text}\n{part}".strip()

                if count_tokens(test_text) > MAX_TOKENS_AMEND and buffer_text:
                    content_str = f"{context_header}\n\n**Nội dung mới{' (tiếp theo)' if sub_idx > 1 else ''}:**\n{buffer_text}"
                    processed.append({
                        "chunk_type": "Amendment_Split",
                        "chunk_id": f"amend_{doc_id}_{tip_id}_part_{sub_idx}",
                        "tip_id": tip_id,
                        "content": content_str.strip(),
                        "bm_title": bm_title,
                        "amended_link": amend.get("amended_link", "")
                    })
                    buffer_text = part
                    sub_idx += 1
                else:
                    buffer_text += f"\n{part}" if buffer_text else part

            if buffer_text:
                content_str = f"{context_header}\n\n**Nội dung mới{' (tiếp theo)' if sub_idx > 1 else ''}:**\n{buffer_text}"
                processed.append({
                    "chunk_type": "Amendment_Split",
                    "chunk_id": f"amend_{doc_id}_{tip_id}_part_{sub_idx}",
                    "tip_id": tip_id,
                    "content": content_str.strip(),
                    "bm_title": bm_title,
                    "amended_link": amend.get("amended_link", "")
                })

    return processed

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
        combined_amends = []
        for d in dieu_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
            if d.get("amends"):
                combined_amends.extend(d["amends"])

        unique_anchors = list(set(combined_anchors))
        unique_amends = list(set(combined_amends))

        chunks.append({
            **base_meta,
            "chunk_type": "Điều",
            "parent_chunk_id": dieu_buffer[0]["parent_id"],
            "content": combined_text,
            "content_embed": clean_embed_content(f"{parent_text}{combined_text}"),
            "chunk_id": chunk_id,
            "anchors": unique_anchors,
            "amends": unique_amends
        })
        dieu_buffer, dieu_tokens = [], 0

    def flush_khoan(parent_id, cau_dan_clean):
        nonlocal khoan_buffer, khoan_tokens, chunks
        if not khoan_buffer: return
        combined_text = "\n\n".join([k["content"] for k in khoan_buffer])
        chunk_id = f"{khoan_buffer[0]['chunk_id']}_gop" if len(khoan_buffer) > 1 else khoan_buffer[0]["chunk_id"]

        combined_anchors = []
        combined_amends = []
        for d in khoan_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
            if d.get("amends"):
                combined_amends.extend(d["amends"])

        unique_anchors = list(set(combined_anchors))
        unique_amends = list(set(combined_amends))

        chunks.append({
            **base_meta,
            "chunk_type": "Khoản",
            "parent_chunk_id": parent_id,
            "content": combined_text,
            "content_embed": clean_embed_content(f"{parent_text}{cau_dan_clean}\n\n{combined_text}"),
            "chunk_id": chunk_id,
            "anchors": unique_anchors,
            "amends": unique_amends
        })
        khoan_buffer, khoan_tokens = [], 0

    def flush_diem(parent_id, cau_dan_clean, k_dan_clean):
        nonlocal diem_buffer, diem_tokens, chunks
        if not diem_buffer: return
        combined_text = "\n".join([d["content"] for d in diem_buffer])
        chunk_id = f"{diem_buffer[0]['chunk_id']}_gop" if len(diem_buffer) > 1 else diem_buffer[0]["chunk_id"]

        combined_anchors = []
        combined_amends = []
        for d in diem_buffer:
            if d.get("anchors"):
                combined_anchors.extend(d["anchors"])
            if d.get("amends"):
                combined_amends.extend(d["amends"])

        unique_anchors = list(set(combined_anchors))
        unique_amends = list(set(combined_amends))

        chunks.append({
            **base_meta,
            "chunk_type": "Điểm",
            "parent_chunk_id": parent_id,
            "content": combined_text,
            "content_embed": clean_embed_content(f"{parent_text}{cau_dan_clean}\n\n{k_dan_clean}\n\n{combined_text}"),
            "chunk_id": chunk_id,
            "anchors": unique_anchors,
            "amends": unique_amends
        })
        diem_buffer, diem_tokens = [], 0

    for i, seg_raw in enumerate(segments):
        raw_cleaned_italic = clean_italic_markers(seg_raw.strip())

        seg_clean, anchors, amends = extract_and_clean_markers(raw_cleaned_italic)
        if not seg_clean or count_tokens(seg_clean) < 10:
            continue

        seg_with_amends = re.sub(r'\[\[ANCHOR:[^\]]+\]\]\s*', '', raw_cleaned_italic).strip()

        seg_id = f"{doc_id}_{anchors[0]}" if anchors else f"{doc_id}_seg_{i}"

        text_for_checking = re.sub(r'\[\[.*?\]\]\s*', '', seg_clean).strip()

        is_phan = re.match(r'^\*(Phần)\s+[^\*]*\*', text_for_checking)
        is_chuong = re.match(r'^\*(Chương)\s+[^\*]*\*', text_for_checking)
        is_muc = re.match(r'^\*(Mục)\s+[^\*]*\*', text_for_checking)

        if is_phan or is_chuong or is_muc:
            flush_dieu()
            if is_phan:
                current_phan_id = current_chuong_id = current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Phần",
                               "parent_chunk_id": doc_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors,
                               "amends": amends})
            elif is_chuong:
                current_chuong_id = current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Chương",
                               "parent_chunk_id": current_phan_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors,
                               "amends": amends})
            elif is_muc:
                current_muc_id = seg_id
                chunks.append({**base_meta, "chunk_type": "Mục",
                               "parent_chunk_id": current_chuong_id,
                               "content": seg_clean, "chunk_id": seg_id, "anchor_ids": anchors,
                               "amends": amends})
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
                     "anchors": anchors,
                     "amends": amends})
                dieu_tokens += tokens
            else:
                flush_dieu()
                khoan_parts_raw = re.split(r'\n(?=(?:Khoản\s+\d+|\d+\.))', seg_with_amends)

                if len(khoan_parts_raw) > 1:
                    cau_dan_clean, _, cau_dan_amends = extract_and_clean_markers(khoan_parts_raw[0])
                    chunks.append({
                        **base_meta,
                        "chunk_type": "Điều",
                        "parent_chunk_id": current_muc_id,
                        "content": cau_dan_clean,
                        "chunk_id": dieu_id,
                        "anchors": anchors,
                        "amends": cau_dan_amends
                    })

                    for j, kp_raw in enumerate(khoan_parts_raw[1:], start=1):
                        k_clean, _, k_amends = extract_and_clean_markers(kp_raw)
                        khoan_id = f"{seg_id}_k{j}"
                        k_tokens = count_tokens(k_clean)

                        if k_tokens <= MAX_TOKENS:
                            if khoan_tokens + k_tokens > MAX_TOKENS:
                                flush_khoan(dieu_id, cau_dan_clean)
                            khoan_buffer.append({"content": k_clean, "chunk_id": khoan_id, "anchors": anchors, "amends": k_amends})
                            khoan_tokens += k_tokens
                        else:
                            flush_khoan(dieu_id, cau_dan_clean)

                            diem_parts_raw = re.split(r'\n(?=[a-zđ]\))', kp_raw)
                            k_dan_clean, _, k_dan_amends = extract_and_clean_markers(diem_parts_raw[0]) if len(
                                diem_parts_raw) > 1 else ("", [], [])
                            chunks.append({**base_meta, "chunk_type": "Khoản",
                                           "parent_chunk_id": dieu_id,
                                           "content": k_dan_clean, "chunk_id": khoan_id, "anchors": anchors,
                                           "amends": k_dan_amends})

                            for k, dp_raw in enumerate(
                                    diem_parts_raw[1:] if len(diem_parts_raw) > 1 else diem_parts_raw):
                                d_clean, _, d_amends = extract_and_clean_markers(dp_raw)
                                diem_id = f"{seg_id}_k{j}_d{k}"

                                d_tokens = count_tokens(d_clean.strip())
                                if d_tokens <= MAX_TOKENS:
                                    if diem_tokens + d_tokens > MAX_TOKENS:
                                        flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                                    diem_buffer.append({
                                        "content": d_clean.strip(),
                                        "chunk_id": diem_id,
                                        "anchors": anchors,
                                        "amends": d_amends
                                    })
                                    diem_tokens += d_tokens
                                else:
                                    flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                                    chunks.append({
                                        **base_meta,
                                        "chunk_type": "Điểm",
                                        "parent_chunk_id": khoan_id,
                                        "content": d_clean.strip(),
                                        "content_embed": clean_embed_content(f"{parent_text}{cau_dan_clean}\n\n{k_dan_clean}\n\n{d_clean}"),
                                        "chunk_id": diem_id,
                                        "anchors": anchors,
                                        "amends": d_amends
                                    })

                            flush_diem(khoan_id, cau_dan_clean, k_dan_clean)
                    flush_khoan(dieu_id, cau_dan_clean)
                else:
                    flush_dieu()
                    para_parts_raw = re.split(r'\n\n+', seg_with_amends)
                    buf_raw, p_idx = "", 1
                    for p_raw in para_parts_raw:
                        test_clean, _, _ = extract_and_clean_markers(buf_raw + "\n\n" + p_raw)
                        if count_tokens(test_clean) > MAX_TOKENS and buf_raw:
                            b_clean, _, b_amends = extract_and_clean_markers(buf_raw)
                            chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                           "parent_chunk_id": current_muc_id,
                                           "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                           "anchors": anchors, "amends": b_amends})
                            buf_raw, p_idx = p_raw, p_idx + 1
                        else:
                            buf_raw = f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw
                    if buf_raw:
                        b_clean, _, b_amends = extract_and_clean_markers(buf_raw)
                        chunks.append({**base_meta, "chunk_type": "Điều_Split",
                                       "parent_chunk_id": current_muc_id,
                                       "content": b_clean, "chunk_id": f"{seg_id}_part{p_idx}",
                                       "anchors": anchors, "amends": b_amends})
        else:
            para_parts_raw = re.split(r'\n\n+', seg_with_amends)
            buf_raw, p_idx = "", 1
            for p_raw in para_parts_raw:
                test_clean, _, _ = extract_and_clean_markers(buf_raw + "\n\n" + p_raw)
                if count_tokens(test_clean) > MAX_TOKENS and buf_raw:
                    b_clean, _, b_amends = extract_and_clean_markers(buf_raw)
                    chunks.append({**base_meta, "chunk_type": "Preamble",
                                   "parent_chunk_id": current_muc_id,
                                   "content": b_clean,
                                   # "content_embed": clean_embed_content(f"{parent_text}{b_clean}"),
                                   "chunk_id": f"{seg_id}_p{p_idx}",
                                   "anchors": anchors,
                                   "amends": b_amends})
                    buf_raw, p_idx = p_raw, p_idx + 1
                else:
                    buf_raw = f"{buf_raw}\n\n{p_raw}".strip() if buf_raw else p_raw
            if buf_raw:
                b_clean, _, b_amends = extract_and_clean_markers(buf_raw)
                chunks.append({**base_meta, "chunk_type": "Preamble",
                               "parent_chunk_id": current_muc_id,
                               "content": b_clean,
                               # "content_embed": clean_embed_content(f"{parent_text}{b_clean}"),
                               "chunk_id": f"{seg_id}_p{p_idx}",
                               "anchors": anchors,
                               "amends": b_amends})

    flush_dieu()

    raw_amendments = json_data.get("amendments", [])

    unique_amendments = []
    seen_amendments = set()

    for item in raw_amendments:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen_amendments:
            seen_amendments.add(item_str)
            unique_amendments.append(item)

    chunks.extend([{**chunk} for chunk in process_amendments(unique_amendments, doc_id)])

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
    OUTPUT_DIR = "data_chunked"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Bắt đầu cắt chunk và lưu vào thư mục: {OUTPUT_DIR}")

    for root, dirs, files in os.walk("data_json_new"):
        for file in files:
            json_file = Path(os.path.join(root, file))
            saved_path = chunk_and_save_file(json_file, OUTPUT_DIR)
            if saved_path:
                print(f"  [+] Đã lưu: {saved_path}")

    for root, dirs, files in os.walk("data_json_next"):
        for file in files:
            json_file = Path(os.path.join(root, file))
            saved_path = chunk_and_save_file(json_file, OUTPUT_DIR)
            if saved_path:
                print(f"  [+] Đã lưu: {saved_path}")