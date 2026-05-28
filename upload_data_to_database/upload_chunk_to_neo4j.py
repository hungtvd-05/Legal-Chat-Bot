import os
import json
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def create_constraints():
    with driver.session() as session:
        session.run("CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        session.run("CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        session.run(
            "CREATE CONSTRAINT unique_sum_root_id IF NOT EXISTS FOR (s:SummaryRoot) REQUIRE s.summary_root_id IS UNIQUE")
        print("[+] Đã kiểm tra/khởi tạo các ràng buộc Unique trên Neo4j.")


def upload_chunks_to_neo4j(chunks: list):
    if not chunks:
        return

    meta_data = next((c for c in chunks if c.get("chunk_type") == "Document_Meta"), {})
    doc_id = meta_data.get("doc_id") or chunks[0].get("doc_id")

    if not doc_id:
        print("  [-] Không tìm thấy doc_id trong dữ liệu chunk, bỏ qua file này.")
        return

    doc_payload = {
        "doc_id": doc_id,
        "title": meta_data.get("title") or meta_data.get("doc_title") or chunks[0].get("doc_title", "Không rõ tiêu đề"),
        "source_url": meta_data.get("source_url", ""),
        "so_hieu": meta_data.get("Số hiệu", ""),
        "loai_van_ban": meta_data.get("Loại văn bản", ""),
        "noi_ban_hang": meta_data.get("Nơi ban hành", ""),
        "nguoi_ky": meta_data.get("Người ký", ""),
        "ngay_ban_hanh": meta_data.get("Ngày ban hành", ""),
        "ngay_hieu_luc": meta_data.get("Ngày hiệu lực", ""),
        "tinh_trang": meta_data.get("Tình trạng", "")
    }

    with driver.session() as session:

        session.run(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d += $doc_payload
            """,
            doc_id=doc_id, doc_payload=doc_payload
        )

        single_summaries = []
        split_summaries = []
        grouped_regular_chunks = defaultdict(list)

        for chunk in chunks:
            c_type = chunk.get("chunk_type")
            if c_type == "Document_Meta":
                continue
            elif c_type == "Summary":
                single_summaries.append(chunk)
            elif c_type == "Summary_Split":
                split_summaries.append(chunk)
            else:
                p_id = chunk.get("parent_chunk_id", doc_id)
                grouped_regular_chunks[p_id].append(chunk)

        for parent_id, chunk_list in grouped_regular_chunks.items():
            prev_chunk_id = None

            for chunk in chunk_list:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue

                payload = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": chunk.get("content", ""),
                    "index": chunk.get("index", 0),
                    "title_layer": chunk.get("title_layer", ""),
                    "chunk_type": chunk.get("chunk_type", "Chunk")
                }

                if parent_id == doc_id or str(parent_id).startswith(doc_id + "_loai") or parent_id == f"meta_{doc_id}":
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c += $payload
                        WITH c
                        MATCH (parent:Document {doc_id: $doc_id})
                        MERGE (c)-[:CHILD_OF]->(parent)
                        """,
                        chunk_id=chunk_id, doc_id=doc_id, payload=payload
                    )
                else:
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c += $payload
                        WITH c
                        MATCH (parent:Chunk {chunk_id: $parent_id})
                        MERGE (c)-[:CHILD_OF]->(parent)
                        """,
                        chunk_id=chunk_id, parent_id=parent_id, payload=payload
                    )

                if prev_chunk_id:
                    session.run(
                        """
                        MATCH (prev:Chunk {chunk_id: $prev_chunk_id})
                        MATCH (curr:Chunk {chunk_id: $curr_chunk_id})
                        MERGE (prev)-[:NEXT_CHUNK]->(curr)
                        """,
                        prev_chunk_id=prev_chunk_id, curr_chunk_id=chunk_id
                    )
                prev_chunk_id = chunk_id

        for sum_chunk in single_summaries:
            chunk_id = sum_chunk.get("chunk_id")
            if not chunk_id:
                continue

            payload = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": sum_chunk.get("content", ""),
                "chunk_type": "Summary"
            }

            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c += $payload
                WITH c
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (c)-[:HAS_SUMMARY]->(d)
                """,
                chunk_id=chunk_id, doc_id=doc_id, payload=payload
            )

        if split_summaries:
            sum_root_id = f"sum_root_{doc_id}"

            session.run(
                """
                MERGE (s:SummaryRoot {summary_root_id: $sum_root_id})
                SET s.doc_id = $doc_id, s.type = "Split_Collection"
                WITH s
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (s)-[:HAS_SUMMARY]->(d)
                """,
                sum_root_id=sum_root_id, doc_id=doc_id
            )

            prev_split_id = None
            for split_chunk in split_summaries:
                chunk_id = split_chunk.get("chunk_id")
                if not chunk_id:
                    continue

                payload = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": split_chunk.get("content", ""),
                    "chunk_type": "Summary_Split"
                }

                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c += $payload
                    WITH c
                    MATCH (s:SummaryRoot {summary_root_id: $sum_root_id})
                    MERGE (c)-[:SUMMARY_PART]->(s)
                    """,
                    chunk_id=chunk_id, sum_root_id=sum_root_id, payload=payload
                )

                if prev_split_id:
                    session.run(
                        """
                        MATCH (prev:Chunk {chunk_id: $prev_split_id})
                        MATCH (curr:Chunk {chunk_id: $curr_split_id})
                        MERGE (prev)-[:NEXT_PART]->(curr)
                        """,
                        prev_split_id=prev_split_id, curr_split_id=chunk_id
                    )
                prev_split_id = chunk_id


if __name__ == "__main__":
    INPUT_DIR = os.getenv("DATA_CHUNKED_DIR")

    print("\n=== BẮT ĐẦU PHASE 3: ĐẨY CẤU TRÚC CHUNK & TÓM TẮT LÊN NEO4J ===")
    create_constraints()

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(INPUT_DIR, filename)
            print(f"   [+] Đang tạo đồ thị cấu trúc cho file: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    chunks_data = json.load(f)
                    if chunks_data:
                        upload_chunks_to_neo4j(chunks_data)
                except json.JSONDecodeError:
                    print(f"   [-] Lỗi định dạng JSON tại file: {filename}")
                except Exception as e:
                    print(f"   [-] Gặp lỗi khi xử lý file {filename}: {str(e)}")

    driver.close()
    print("\n=== HOÀN THÀNH: Đã đồng bộ toàn bộ các định dạng chunk lên Neo4j! ===")