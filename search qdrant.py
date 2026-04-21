import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import time

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
RERANKER_MODEL_NAME = "AITeamVN/Vietnamese_Reranker"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "luat_giao_thong"


class LegalRAGPipeline:
    def __init__(self):
        print("[1] Đang tải Embedding Model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("[2] Đang tải Reranker Model...")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        print("[3] Kết nối Qdrant...")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

        print("=> Khởi tạo Legal RAG Pipeline thành công!\n")

    def _call_gemini(self, prompt: str, temperature: float = 0.3) -> str:
        """Hàm helper gọi Gemini có tính năng tự động thử lại nếu server bận"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2048,
                    )
                )
                return response.text.strip()
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "UNAVAILABLE" in error_msg:
                    print(
                        f"Server Gemini đang bận (lần thử {attempt + 1}/{max_retries}). Đang chờ 3 giây để thử lại...")
                    time.sleep(3)
                else:
                    print(f"Lỗi khi gọi Gemini: {e}")
                    return "Lỗi hệ thống khi sinh câu trả lời."

        return "Hệ thống Gemini đang quá tải, vui lòng thử lại sau."

    def expand_query_with_llm(self, original_query: str) -> list[str]:
        """Mở rộng truy vấn bằng LLM (HyDE-style)"""
        prompt = f"""
Bạn là chuyên gia pháp luật Việt Nam.
Người dùng hỏi: "{original_query}"

Hãy tạo ra **3 câu truy vấn tìm kiếm** khác nhau (viết dưới dạng câu khẳng định hoặc từ khóa pháp lý) để tìm trong cơ sở dữ liệu vector.
Chỉ trả về mỗi truy vấn trên một dòng, không giải thích thêm.
"""

        try:
            generated_text = self._call_gemini(prompt, temperature=0.4)
            generated_queries = [line.strip("- •* ") for line in generated_text.split('\n') if line.strip()]
            queries = [original_query] + generated_queries
            return queries[:4]
        except Exception as e:
            print(f"Lỗi mở rộng truy vấn: {e}")
            return [original_query]

    def retrieve_from_qdrant(self, queries: list[str], top_k_per_query: int = 6) -> list[dict]:
        """Tìm kiếm vector + deduplicate"""
        unique_chunks = {}

        for query in queries:
            query_vector = self.embed_model.encode(query).tolist()

            search_response = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k_per_query,
                with_payload=True
            )

            for hit in search_response.points:
                if hit.id not in unique_chunks:
                    unique_chunks[hit.id] = hit.payload

        return list(unique_chunks.values())

    def rerank_chunks(self, query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
        if not chunks:
            return []

        pairs = [[query, chunk.get("content", "")] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]

    def generate_final_answer(self, query: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            return "Không tìm thấy tài liệu liên quan để trả lời câu hỏi này."

        context_text = ""
        for i, chunk in enumerate(context_chunks):
            title = chunk.get("title", "Tài liệu không tên")
            content = chunk.get("content", "")
            context_text += f"\n--- TÀI LIỆU {i + 1}: {title} ---\n{content}\n"

        prompt = f"""
Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp, trung thực và cẩn thận.

Câu hỏi của người dùng: "{query}"

Tài liệu tham khảo:
{context_text}

Yêu cầu nghiêm ngặt:
1. Chỉ trả lời dựa trên tài liệu được cung cấp, không bịa thông tin.
2. Nếu không đủ thông tin → nói rõ "Thông tin hiện tại chưa đủ để trả lời chính xác."
3. Trích dẫn rõ Điều, Khoản, Luật nào (nếu có).
4. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu, dùng gạch đầu dòng khi cần.

Trả lời:
"""

        return self._call_gemini(prompt, temperature=0.2)

    def ask(self, query: str):
        print(f"\n[❓] Câu hỏi: {query}")

        print("Đang mở rộng truy vấn...")
        expanded_queries = self.expand_query_with_llm(query)
        print(f"   → Truy vấn: {expanded_queries}")

        print("Đang tìm kiếm trong cơ sở dữ liệu...")
        retrieved_chunks = self.retrieve_from_qdrant(expanded_queries, top_k_per_query=6)
        print(f"   → Tìm thấy {len(retrieved_chunks)} chunks.")

        print("Đang reranking...")
        top_chunks = self.rerank_chunks(query, retrieved_chunks, top_n=4)

        print("\nTop chunks liên quan nhất:")
        for idx, chunk in enumerate(top_chunks):
            short = chunk.get("content", "")[:120].replace("\n", " ")
            print(f"   [{idx + 1}] Score: {chunk['rerank_score']:.4f} | {short}...")

        print("\nĐang sinh câu trả lời...")
        answer = self.generate_final_answer(query, top_chunks)

        print("\n" + "=" * 70)
        print("TRẢ LỜI TỪ AI LUẬT SƯ:")
        print("=" * 70)
        print(answer)
        print("=" * 70)


if __name__ == "__main__":
    pipeline = LegalRAGPipeline()

    print("Hệ thống Legal RAG sẵn sàng. Nhập 'exit' để thoát.\n")

    while True:
        user_input = input("Nhập câu hỏi pháp lý: ").strip()
        if user_input.lower() in ['exit', 'quit', 'thoát']:
            print("Tạm biệt!")
            break
        if user_input:
            pipeline.ask(user_input)