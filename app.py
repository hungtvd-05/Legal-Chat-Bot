import os
import re
import time
import itertools
import math
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

load_dotenv()

LM_MODEL = "qwen/qwen3-8b"

lm_clients = [
    OpenAI(base_url="https://zus-lucas-ready-powder.trycloudflare.com/v1", api_key="lm-studio-gpu1"),
]
client_iterator = itertools.cycle(lm_clients)

EMBEDDING_MODEL_NAME = "./nrk-legal-large-traffic-ft-merged-v2"
RERANKER_MODEL_NAME = "AITeamVN/Vietnamese_Reranker"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "luat_giao_thong_new_finetune_model"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class LegalRAGPipeline:
    def __init__(self):
        print("[+] Đang khởi tạo Embedding Model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("[+] Đang khởi tạo Reranker Model...")
        self.rerank_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=1024)

        print("[+] Đang kết nối Qdrant...")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

        print("[+] Đang kết nối Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        self.build_local_bm25()

    def build_local_bm25(self):
        print("[+] Khởi tạo bộ nhớ lập chỉ mục từ khóa BM25...")
        try:
            scroll_results = self.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            points = scroll_results.points if hasattr(scroll_results, 'points') else scroll_results[0]

            self.bm25_corpus = []
            self.bm25_mapping = []

            for p in points:
                payload = p.payload
                content = payload.get("content", "")
                if content:
                    self.bm25_corpus.append(content.lower().split())
                    self.bm25_mapping.append(payload)

            if self.bm25_corpus:
                self.bm25 = BM25Okapi(self.bm25_corpus)
            else:
                self.bm25 = None
        except Exception as e:
            print(f"  [-] Không thể build BM25: {e}")
            self.bm25 = None

    def get_llm_client(self) -> OpenAI:
        return next(client_iterator)

    def _call_llm_sync(self, prompt: str, temperature: float = 0.2) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            client = self.get_llm_client()
            try:
                response = client.chat.completions.create(
                    model=LM_MODEL,
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý pháp luật chuyên nghiệp. Hãy thực hiện chính xác yêu cầu của người dùng."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=200
                )
                raw_text = response.choices[0].message.content.strip()
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                return clean_text
            except Exception as e:
                print(f"    [LLM] Lỗi gọi LLM sinh từ khóa (lần {attempt + 1}): {e}")
                time.sleep(2)
        return ""

    def expand_query_with_llm(self, original_query: str) -> list[str]:
        prompt = f"""Bạn là một chuyên gia phân tích từ khóa pháp lý. 
Nhiệm vụ của bạn là trích xuất và tạo ra 2 biến thể tìm kiếm khác nhau dựa trên câu hỏi gốc để quét cơ sở dữ liệu luật.

QUY TẮC NGHIÊM NGẶT:
1. GIỮ NGUYÊN các thuật ngữ cốt lõi như: "xe cơ quan", "tái xuất khẩu", "hồ sơ".
2. KHÔNG tự ý đổi "xe cơ quan" thành "xe công vụ", "xe công", hoặc "xe biển xanh".
3. KHÔNG thêm các từ ngữ không có trong ngữ cảnh như "tái nhập cảnh", "biên giới", "hải quan".
4. Chỉ sinh thêm các từ khóa bổ trợ liên quan trực tiếp như: "thành phần hồ sơ", "thủ tục", "giấy tờ".
5. CHỈ TRẢ VỀ các câu tìm kiếm, mỗi câu một dòng. KHÔNG viết số thứ tự, không giải thích.

Câu hỏi gốc: "{original_query}"

Biến thể tìm kiếm (Mỗi câu một dòng):"""

        generated_text = self._call_llm_sync(prompt, temperature=0.2)
        if not generated_text:
            return [original_query]

        queries = [original_query]
        for line in generated_text.split('\n'):
            clean_line = line.strip("- •* 1234567890. ")
            if clean_line and len(clean_line) > 10:
                queries.append(clean_line)
        return queries[:3]

    def search_qdrant_dense(self, queries: list[str], top_k: int = 8):
        unique_chunks = {}
        for query in queries:
            query_vector = self.embed_model.encode(query).tolist()
            response = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            for hit in response.points:
                chunk_id = hit.payload.get("chunk_id")
                if chunk_id and chunk_id not in unique_chunks:
                    unique_chunks[chunk_id] = hit.payload
        return list(unique_chunks.values())

    def search_bm25_sparse(self, query: str, top_k: int = 12):
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.bm25_mapping[i] for i in top_indices if scores[i] > 0]

    def hybrid_rerank(self, query: str, chunks: list, top_n: int = 4):
        if not chunks:
            return []

        seen = set()
        unique_chunks = []
        for c in chunks:
            c_id = c.get("chunk_id")
            if c_id not in seen:
                seen.add(c_id)
                unique_chunks.append(c)

        tokenized_query = query.lower().split()
        tokenized_chunks = [c.get("content", "").lower().split() for c in unique_chunks]
        bm25_local = BM25Okapi(tokenized_chunks)
        raw_bm25_scores = bm25_local.get_scores(tokenized_query)

        min_bm25, max_bm25 = min(raw_bm25_scores), max(raw_bm25_scores)
        if max_bm25 > min_bm25:
            norm_bm25_scores = [(score - min_bm25) / (max_bm25 - min_bm25) for score in raw_bm25_scores]
        else:
            norm_bm25_scores = [1.0] * len(raw_bm25_scores)

        pairs = [[query, c.get("content", "")] for c in unique_chunks]
        raw_rerank_scores = self.rerank_model.predict(pairs)

        if hasattr(raw_rerank_scores, "tolist"):
            raw_rerank_scores = raw_rerank_scores.tolist()
        if isinstance(raw_rerank_scores, list) and len(raw_rerank_scores) > 0 and isinstance(raw_rerank_scores[0], list):
            raw_rerank_scores = [s[0] for s in raw_rerank_scores]

        raw_rerank_scores = [float(s) for s in raw_rerank_scores]
        min_rerank, max_rerank = min(raw_rerank_scores), max(raw_rerank_scores)
        if max_rerank > min_rerank:
            norm_rerank_scores = [(score - min_rerank) / (max_rerank - min_rerank) for score in raw_rerank_scores]
        else:
            norm_rerank_scores = [1.0] * len(raw_rerank_scores)

        for i, chunk in enumerate(unique_chunks):
            chunk["hybrid_score"] = (0.3 * norm_bm25_scores[i]) + (0.7 * norm_rerank_scores[i])

        unique_chunks = sorted(unique_chunks, key=lambda x: x["hybrid_score"], reverse=True)
        return unique_chunks[:top_n]

    def get_pure_document_node(self, doc_id: str):
        with self.neo4j_driver.session() as session:
            query = """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN properties(d) AS props
            """
            result = session.run(query, doc_id=doc_id).single()
            if result and result["props"]:
                props = result["props"]
                meta_str = "=== THÔNG TIN HÀNH CHÍNH (METADATA) GỐC ===\n"
                for key, val in props.items():
                    if val and key != "doc_id":
                        meta_str += f"- {key}: {val}\n"
                return meta_str
        return None

    def get_sliding_window_summary_split(self, chunk_id: str):
        with self.neo4j_driver.session() as session:
            query = """
                MATCH (curr:Chunk {chunk_id: $chunk_id})
                OPTIONAL MATCH (prev:Chunk)-[:NEXT_PART]->(curr)
                OPTIONAL MATCH (curr)-[:NEXT_PART]->(next:Chunk)
                RETURN prev.content AS p_content, curr.content AS c_content, next.content AS n_content
            """
            result = session.run(query, chunk_id=chunk_id).single()
            if result:
                summary_parts = []
                if result["p_content"]:
                    summary_parts.append(f"[Phần trước]: {result['p_content']}")
                if result["c_content"]:
                    summary_parts.append(f"[Phần hiện tại]: {result['c_content']}")
                if result["n_content"]:
                    summary_parts.append(f"[Phần kế tiếp]: {result['n_content']}")
                return "\n\n".join(summary_parts)
        return None

    def get_graph_context_from_neo4j(self, chunk_id: str):
        context_extension = {"parent_info": None, "next_chunk_content": None}
        with self.neo4j_driver.session() as session:
            parent_query = """
                MATCH (c:Chunk {chunk_id: $chunk_id})-[:CHILD_OF]->(p)
                RETURN labels(p) AS labels, p.content AS content, p.title AS title
            """
            res_p = session.run(parent_query, chunk_id=chunk_id).single()
            if res_p:
                if "Document" in res_p["labels"]:
                    context_extension["parent_info"] = f"Văn bản gốc: {res_p['title']}"
                else:
                    clean_text = res_p['content'].split('\n')[0] if res_p['content'] else "Không rõ"
                    context_extension["parent_info"] = f"Thuộc cấp cha: {clean_text}"

            next_query = """
                MATCH (c:Chunk {chunk_id: $chunk_id})-[:NEXT_CHUNK]->(next:Chunk)
                RETURN next.content AS content
            """
            res_n = session.run(next_query, chunk_id=chunk_id).single()
            if res_n:
                context_extension["next_chunk_content"] = res_n["content"]
        return context_extension

    def generate_final_answer_stream(self, query: str, final_contexts: list):
        client = self.get_llm_client()
        context_text = "\n\n".join(final_contexts)

        system_prompt = (
            "Bạn là trợ lý ảo chuyên về Luật giao thông đường bộ Việt Nam. "
            "Trả lời chính xác dựa trên ngữ cảnh pháp lý được cung cấp, trích dẫn Điều/Khoản cụ thể. "
            "Tuyệt đối không bịa đặt thông tin ngoài tài liệu.\n\n"
            "ĐỊNH DẠNG ĐẦU RA (BẮT BUỘC TUÂN THỦ CHÍNH XÁC):\n"
            "1. Mở đầu: 1–2 câu tổng quan ngắn, KHÔNG có tiêu đề.\n"
            "2. Tiêu đề mục: dùng ## cho mục chính, ### cho mục phụ.\n"
            "   - LUÔN có một DÒNG TRỐNG trước ## và sau ##.\n"
            "   - KHÔNG viết nội dung tiếp theo ngay sau ## trên cùng một dòng.\n"
            "3. Danh sách: dùng '- ' (gạch ngang + dấu cách), KHÔNG dùng '*' hay '•'.\n"
            "   - Mỗi mục trên một dòng riêng biệt.\n"
            "   - Có một DÒNG TRỐNG trước mục đầu tiên của danh sách.\n"
            "4. In đậm: dùng *văn bản* cho tên luật, số điều khoản, mức phạt.\n"
            "5. Kết thúc: viết dòng '---' rồi xuống dòng '*Kết luận:* [tóm tắt 1–2 câu]'.\n\n"
            "VÍ DỤ CẤU TRÚC ĐÚNG (làm theo mẫu này):\n"
            "```\n"
            "Theo quy định hiện hành, xe máy vượt đèn đỏ bị xử phạt hành chính.\n\n"
            "## 1. Căn cứ pháp lý\n\n"
            "Theo *Điều 7, Khoản 4* Nghị định 123/2025/NĐ-CP, mức phạt áp dụng:\n\n"
            "- Phạt tiền từ **800.000đ đến 1.200.000đ**\n"
            "- Tước quyền sử dụng GPLX từ 1 đến 3 tháng (nếu tái phạm)\n\n"
            "## 2. Lưu ý bổ sung\n\n"
            "Mức phạt có thể tăng nếu có tình tiết tăng nặng theo *Điều 10*.\n\n"
            "---\n\n"
            "*Kết luận:* Vượt đèn đỏ bị phạt từ 800.000đ đến 1.200.000đ và có thể bị tước GPLX.\n"
            "```\n"
        )
        user_prompt = f"""Dưới đây là ngữ cảnh pháp lý và cấu trúc phân cấp đồ thị đã được tối ưu liên quan đến câu hỏi:
{context_text}

Câu hỏi người dùng: {query}

Hãy phân tích kỹ lưuỡng dữ liệu trên và trả lời chi tiết:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return client.chat.completions.create(
            model=LM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
            stream=True
        )

pipeline_holder = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SYSTEM] Đang khởi tạo hệ thống GraphRAG Pipeline...")
    pipeline_holder["pipeline"] = LegalRAGPipeline()
    print("[SYSTEM] Hệ thống đã sẵn sàng xử lý API request!")
    yield
    print("[SYSTEM] Đang giải phóng kết nối cơ sở dữ liệu...")
    if "pipeline" in pipeline_holder:
        pipeline_holder["pipeline"].neo4j_driver.close()


app = FastAPI(
    title="Luật Giao Thông Việt Nam GraphRAG API",
    description="Hệ thống API truy vấn luật giao thông đường bộ sử dụng kỹ thuật Hybrid RAG và Graph Tri thức",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: QueryRequest):
    pipeline: LegalRAGPipeline = pipeline_holder.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Hệ thống chưa khởi tạo xong.")

    def stream_generator():
        query_text = request.query
        print(f"\n[API API] Nhận yêu cầu: '{query_text}'")

        expanded_queries = pipeline.expand_query_with_llm(query_text)
        dense_results = pipeline.search_qdrant_dense(expanded_queries, top_k=8)
        sparse_results = pipeline.search_bm25_sparse(query_text, top_k=12)
        all_retrieved = dense_results + sparse_results

        top_chunks = pipeline.hybrid_rerank(query_text, all_retrieved, top_n=4)

        final_contexts = []
        processed_summary_chunks = set()

        for idx, chunk in enumerate(top_chunks):
            c_type = chunk.get("chunk_type")
            d_id = chunk.get("doc_id")
            c_id = chunk.get("chunk_id")

            if c_type == "Document_Meta" and d_id:
                meta_context = pipeline.get_pure_document_node(d_id)
                if meta_context:
                    final_contexts.append(meta_context)
            elif c_type == "Summary_Split" and c_id:
                if c_id not in processed_summary_chunks:
                    sliding_summary = pipeline.get_sliding_window_summary_split(c_id)
                    if sliding_summary:
                        final_contexts.append(f"=== PHÂN ĐOẠN TÓM TẮT VĂN BẢN ===\n{sliding_summary}")
                        processed_summary_chunks.add(c_id)
            else:
                graph_ext = pipeline.get_graph_context_from_neo4j(c_id) if c_id else {}
                ctx_str = f"--- Phân đoạn ngữ cảnh {idx + 1} ---\n"
                if graph_ext.get("parent_info"):
                    ctx_str += f"[{graph_ext['parent_info']}]\n"
                ctx_str += f"Nội dung luật: {chunk.get('content', '')}\n"
                if graph_ext.get("next_chunk_content"):
                    ctx_str += f"Nội dung bổ trợ liền sau: {graph_ext['next_chunk_content']}\n"
                final_contexts.append(ctx_str)

        response_stream = pipeline.generate_final_answer_stream(query_text, final_contexts)

        inside_think_tag = False
        for chunk in response_stream:
            if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                if "<think>" in content:
                    inside_think_tag = True
                    continue
                if "</think>" in content:
                    inside_think_tag = False
                    continue

                if not inside_think_tag:
                    yield f"data: {content}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)