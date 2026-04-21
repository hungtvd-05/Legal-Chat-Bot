import json
import re
import os
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase
from dotenv import load_dotenv
import time

load_dotenv()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

from openai import OpenAI

LM_MODEL = "qwen/qwen3.5-4b"

lm_clients = [
    OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio-gpu1"),
    OpenAI(base_url="https://uri-christopher-successfully-raising.trycloudflare.com/v1", api_key="lm-studio-gpu2")
]
client_iterator = itertools.cycle(lm_clients)
MAX_WORKERS = 4

def extract_legal_entities(content: str, client: OpenAI):
    prompt = f"""Phân tích đoạn văn bản luật sau và trích xuất các thực thể pháp lý theo định dạng JSON:
Nội dung: "{content}"

Yêu cầu trích xuất:
1. Chủ thể (Subject): Ai là người thực hiện, bị phạt, hoặc chịu tác động?
2. Hành vi (Action): Hành vi nào được quy định, cấm, hoặc vi phạm?
3. Chế tài (Penalty): Hình phạt, mức phạt hoặc hệ quả pháp lý là gì?

Chỉ trả về JSON hợp lệ, không giải thích. Nếu không có thì để mảng rỗng [].
Ví dụ: {{"subjects": ["Người điều khiển xe mô tô"], "actions": ["Vượt đèn đỏ"], "penalties": ["Phạt tiền từ 600.000 đến 1.000.000 đồng"]}}
/no_think"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là công cụ trích xuất thực thể pháp lý. Chỉ trả về JSON, không giải thích."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            clean_json = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            print(f"    [LM] Lỗi attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


def upload_to_neo4j_and_extract(chunks: list):
    with neo4j_driver.session() as session:
        meta_chunk = next((c for c in chunks if c.get("chunk_type") == "Document_Meta"), chunks[0])
        doc_id = meta_chunk.get('doc_id')

        session.run("""
                                MERGE (d:Document {doc_id: $doc_id})
                                SET d.title = $title,
                                    d.source_url = $source_url,
                                    d.loai_van_ban = $loai_van_ban,
                                    d.noi_ban_hanh = $noi_ban_hanh,
                                    d.nguoi_ky = $nguoi_ky,
                                    d.ngay_ban_hanh = $ngay_ban_hanh,
                                    d.ngay_hieu_luc = $ngay_hieu_luc,
                                    d.ngay_cong_bao = $ngay_cong_bao,
                                    d.so_cong_bao = $so_cong_bao,
                                    d.tinh_trang = $tinh_trang
                            """,
                    doc_id=doc_id,
                    title=meta_chunk.get('title'),
                    source_url=meta_chunk.get('source_url'),
                    loai_van_ban=meta_chunk.get('Loại văn bản'),
                    noi_ban_hanh=meta_chunk.get('Nơi ban hành'),
                    nguoi_ky=meta_chunk.get('Người ký'),
                    ngay_ban_hanh=meta_chunk.get('Ngày ban hành'),
                    ngay_hieu_luc=meta_chunk.get('Ngày hiệu lực'),
                    ngay_cong_bao=meta_chunk.get('Ngày công báo'),
                    so_cong_bao=meta_chunk.get('Số công báo'),
                    tinh_trang=meta_chunk.get('Tình trạng')
                    )

        for chunk in chunks:
            full_name = chunk['content'].split('\n')[0].strip() if chunk.get('chunk_type') in ['Amendment',
                                                                                               'Amendment_Split'] else \
                chunk.get('content_embed', chunk['content']).split('\n')[0].strip()

            session.run("""
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            SET c.content = $content,
                                c.chunk_type = $chunk_type,
                                c.doc_id = $doc_id,
                                c.name = $full_name,
                                c.parent_chunk_id = $parent_chunk_id,
                                c.bm_title = $bm_title,
                                c.amended_link = $amended_link
                        """,
                        chunk_id=chunk['chunk_id'], content=chunk['content'], chunk_type=chunk.get('chunk_type'),
                        doc_id=doc_id, full_name=full_name, parent_chunk_id=chunk.get('parent_chunk_id'),
                        bm_title=chunk.get('bm_title'), amended_link=chunk.get('amended_link'))

        for chunk in chunks:
            chunk_type = chunk.get('chunk_type')
            chunk_id = chunk['chunk_id']
            parent_chunk_id = chunk.get('parent_chunk_id')

            if chunk_type in ['Amendment', 'Amendment_Split']:
                tip_id = chunk.get('tip_id')
                amended_link = chunk.get('amended_link')

                if tip_id:
                    target_chunk_ids = [c['chunk_id'] for c in chunks if 'amends' in c and tip_id in c['amends']]
                    if target_chunk_ids:
                        session.run("""
                            UNWIND $target_ids AS target_id
                            MATCH (target:Chunk {chunk_id: target_id})
                            MATCH (amend:Chunk {chunk_id: $chunk_id})
                            MERGE (target)-[:HAS_AMENDMENT]->(amend)
                        """, target_ids=target_chunk_ids, chunk_id=chunk_id)

                if amended_link:
                    base_url = amended_link.split('?')[0].split('#')[0]
                    session.run("""
                        MATCH (amend:Chunk {chunk_id: $chunk_id})
                        MATCH (d:Document {source_url: $base_url})
                        MERGE (amend)-[:AMENDS_DOCUMENT]->(d)
                    """, chunk_id=chunk_id, base_url=base_url)
                continue

            if chunk_type == 'Document_Meta':
                session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:HAS_METADATA]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)
                continue

            if chunk_type in ['Summary', 'Summary_Split']:
                session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:HAS_SUMMARY]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)
                continue

            if parent_chunk_id == doc_id:
                session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)
            elif parent_chunk_id:
                session.run("""
                        MATCH (p:Chunk {chunk_id: $parent_chunk_id, doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (p)-[:CONTAINS]->(c)
                    """, parent_chunk_id=parent_chunk_id, chunk_id=chunk_id, doc_id=doc_id)

    parent_ids = {c.get("parent_chunk_id") for c in chunks if c.get("parent_chunk_id")}
    skip_llm_types = ['Document_Meta', 'Summary', 'Summary_Split', 'Phần', 'Chương', 'Mục', 'Preamble']

    chunks_to_process = []
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type")
        if chunk_type in skip_llm_types:
            continue

        is_amendment = chunk_type in ['Amendment', 'Amendment_Split']
        is_leaf = chunk["chunk_id"] not in parent_ids

        if is_amendment or is_leaf:
            text_for_llm = chunk.get("content_embed", chunk["content"]) if not is_amendment else chunk["content"]
            chunks_to_process.append({
                "chunk_id": chunk["chunk_id"],
                "text": text_for_llm
            })

    print(f"    [*] Đang bóc tách thực thể cho {len(chunks_to_process)} chunks bằng 2 GPUs...")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {}
        for item in chunks_to_process:
            client = next(client_iterator)

            future = executor.submit(extract_legal_entities, item["text"], client)
            future_to_chunk[future] = item["chunk_id"]

        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                entities = future.result()
                if entities:
                    results.append({
                        "chunk_id": chunk_id,
                        "entities": entities
                    })
            except Exception as exc:
                print(f"    [!] Chunk {chunk_id} sinh ra lỗi: {exc}")

    extracted_count = 0
    if results:
        with neo4j_driver.session() as session:
            for res in results:
                chunk_id = res["chunk_id"]
                entities = res["entities"]

                success = False
                for entity_type, items in entities.items():
                    if not items or not isinstance(items, list):
                        continue

                    label = entity_type.capitalize()[:-1]
                    for item in items:
                        item = str(item).strip()
                        if not item: continue

                        success = True
                        session.run(f"""
                                MATCH (c:Chunk {{chunk_id: $chunk_id}})
                                MERGE (e:{label} {{name: $item_name}})
                                MERGE (c)-[:MENTIONS]->(e)
                            """, chunk_id=chunk_id, item_name=item)

                if success:
                    extracted_count += 1

    print(f"    + Đã bóc tách và lưu xong Tri thức cho {extracted_count} nút lá & sửa đổi.")

if __name__ == "__main__":
    INPUT_DIR = "data_chunked_new"

    print("\n=== BẮT ĐẦU PHASE 3: TẠO GRAPH ===")

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(INPUT_DIR, filename)
            print(f"  [+] Đang tạo GRAPH và đẩy DB cho: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            if chunks_data:
                upload_to_neo4j_and_extract(chunks_data)