import json
import os
import random
import time
import re
import itertools
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

LM_MODEL = "qwen/qwen3.5-4b"

lm_clients = [
    OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio-gpu1"),
]

MAX_WORKERS       = 12
WRITE_BUFFER_SIZE = 50
PRINT_EVERY_N     = 20
MIN_WORDS         = 50

_client_lock     = threading.Lock()
_client_iterator = itertools.cycle(lm_clients)

def get_next_client() -> OpenAI:
    with _client_lock:
        return next(_client_iterator)

_write_queue: queue.Queue = queue.Queue()
_SENTINEL = object()

def _writer_thread(output_file: str):
    buffer = []
    with open(output_file, 'a', encoding='utf-8') as f:
        while True:
            item = _write_queue.get()

            if item is _SENTINEL:
                if buffer:
                    f.writelines(buffer)
                    f.flush()
                break

            buffer.append(item)

            if len(buffer) >= WRITE_BUFFER_SIZE:
                f.writelines(buffer)
                f.flush()
                buffer.clear()

def enqueue_triplet(triplet: dict):
    line = json.dumps(triplet, ensure_ascii=False) + '\n'
    _write_queue.put(line)

def generate_queries_from_llm(content: str, client: OpenAI) -> list:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là chuyên gia tạo synthetic query từ văn bản pháp luật.
QUY TẮC NGHIÊM NGẶT:
1. Tạo tối đa 2 câu hỏi tra cứu thực tế, độc lập, rõ ràng.
2. NGÔN NGỮ: BẮT BUỘC 100% BẰNG TIẾNG VIỆT. TUYỆT ĐỐI KHÔNG SỬ DỤNG TIẾNG TRUNG QUỐC HAY TIẾNG ANH.
3. CẤM dùng đại từ phiếm chỉ ("luật này", "điều này", "văn bản trên"). Phải dùng đích danh tên luật hoặc chủ thể.
4. Nếu văn bản là tiêu đề, mục lục hoặc không đủ ý → BẮT BUỘC trả về mảng rỗng [].
5. CHỈ xuất ra định dạng JSON Array chứa các chuỗi string. Không giải thích, không thêm văn bản.

VÍ DỤ ĐẦU RA KỲ VỌNG:
["Mức phạt khi không đội mũ bảo hiểm là bao nhiêu?", "Cơ quan nào có thẩm quyền cấp giấy phép lái xe?"]
Hoặc nếu không đủ ý:
[]"""
                    },
                    {
                        "role": "user",
                        "content": f"Đoạn văn bản:\n{content}\n\nTrả về JSON Array:"
                    }
                ],
                temperature=0.5,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            clean_json = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)

        except json.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            print(f"\n    [LM] Lỗi attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(3)
    return []

FORBIDDEN_WORDS = ["luật này", "văn bản này", "điều này", "khoản này", "dưới đây"]

def process_single_chunk(
    pos_chunk:     dict,
    chunks_by_doc: dict,
    all_eligible:  list,
) -> int:
    positive_text = pos_chunk["content_embed"]
    doc_id        = pos_chunk["doc_id"]
    chunk_id      = pos_chunk["chunk_id"]

    client  = get_next_client()
    queries = generate_queries_from_llm(positive_text, client)
    if not queries:
        return 0

    same_doc_pool = [c for c in chunks_by_doc.get(doc_id, []) if c["chunk_id"] != chunk_id]
    hard_neg = random.choice(same_doc_pool)["content_embed"] if same_doc_pool else ""

    easy_neg = ""
    for _ in range(30):
        candidate = random.choice(all_eligible)
        if candidate["doc_id"] != doc_id:
            easy_neg = candidate["content_embed"]
            break

    count = 0
    for q in queries:
        q = q.strip()
        if len(q) < 10:
            continue
        if re.search(r'[\u4e00-\u9fff]', q):
            print(f"\n[!] Lọc tiếng Trung: {q}")
            continue
        if any(bad in q.lower() for bad in FORBIDDEN_WORDS):
            print(f"\n[!] Lọc phiếm chỉ: {q}")
            continue

        triplet: dict = {"query": q, "pos": [positive_text], "neg": []}
        if hard_neg: triplet["neg"].append(hard_neg)
        if easy_neg: triplet["neg"].append(easy_neg)

        enqueue_triplet(triplet)
        count += 1

    return count


def build_triplets_dataset(input_dir: str, output_file: str):

    all_chunks: list = []
    print("1. Đang đọc dữ liệu JSON gốc...")
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_chunks.extend(c for c in data if "content_embed" in c)
    print(f"   Tổng số chunk hợp lệ: {len(all_chunks)}")

    processed_texts: set = set()
    if os.path.exists(output_file):
        print(f"2. Tìm thấy {output_file}. Đang đọc tiến độ cũ...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed_texts.add(json.loads(line)["pos"][0])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"   Đã xử lý trước đó: {len(processed_texts)} chunk")
    else:
        print("2. Chưa có file output, bắt đầu từ đầu.")

    chunks_to_process = [c for c in all_chunks if c["content_embed"] not in processed_texts]
    print(f"   Còn lại cần xử lý: {len(chunks_to_process)} chunk")

    if not chunks_to_process:
        print("[Hoàn tất] Toàn bộ dữ liệu đã được xử lý!")
        return

    print("3. Pre-computing negative pools...")

    for c in all_chunks:
        c["_wc"] = len(c["content_embed"].split())

    all_eligible: list = [c for c in all_chunks if c["_wc"] >= MIN_WORDS]

    from collections import defaultdict
    chunks_by_doc: dict = defaultdict(list)
    for c in all_eligible:
        chunks_by_doc[c["doc_id"]].append(c)

    print(f"   Eligible chunks (>= {MIN_WORDS} words): {len(all_eligible)}")
    print(f"   Unique docs: {len(chunks_by_doc)}")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    writer = threading.Thread(target=_writer_thread, args=(output_file,), daemon=True)
    writer.start()
    print(f"\n4. Writer thread khởi động → {output_file}")

    print(f"5. Bắt đầu xử lý với {MAX_WORKERS} worker threads...\n")

    total_chunks   = len(chunks_to_process)
    done_chunks    = 0
    total_triplets = 0
    error_count    = 0
    _counter_lock  = threading.Lock()

    def print_progress():
        pct = done_chunks / total_chunks * 100 if total_chunks else 0
        print(
            f"\r  Chunk: {done_chunks:>6}/{total_chunks} ({pct:5.1f}%) | "
            f"Triplets: {total_triplets:>8} | "
            f"Queue: {_write_queue.qsize():>4} | "
            f"Lỗi: {error_count}   ",
            end="", flush=True
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_single_chunk, chunk, chunks_by_doc, all_eligible): chunk
            for chunk in chunks_to_process
        }

        for future in as_completed(future_map):
            with _counter_lock:
                done_chunks += 1
                try:
                    total_triplets += future.result()
                except Exception as exc:
                    error_count += 1
                    print(f"\n[-] Lỗi chunk: {exc}")

                if done_chunks % PRINT_EVERY_N == 0 or done_chunks == total_chunks:
                    print_progress()

    print(f"\n\n6. Flushing queue ({_write_queue.qsize()} items còn lại)...")
    _write_queue.put(_SENTINEL)
    writer.join()

    print(f"\n[✓ Hoàn tất] Đã lưu {total_triplets} triplets → '{output_file}'")
    print(f"   Chunks xử lý: {done_chunks} | Lỗi: {error_count}")


if __name__ == "__main__":
    INPUT_DIR   = os.getenv("DATA_CHUNKED_DIR")
    OUTPUT_FILE = os.getenv("TRAIN_DATASET")

    build_triplets_dataset(INPUT_DIR, OUTPUT_FILE)