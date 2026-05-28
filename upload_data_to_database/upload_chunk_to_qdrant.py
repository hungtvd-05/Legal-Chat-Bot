import json
import uuid
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "./nrk-legal-large-traffic-ft-merged-v2"

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

COLLECTION_NAME = "luat_giao_thong_new_finetune_model"
QDRANT_URL = "http://localhost:6333"
qdrant_client = QdrantClient(url=QDRANT_URL)
try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    print(f"  -> Collection '{COLLECTION_NAME}' chưa tồn tại trên Qdrant. Đang tạo mới...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )

def upload_to_qdrant(chunks: list):
    filtered_chunks = [
        chunk for chunk in chunks
        if chunk.get("content_embed") and str(chunk["content_embed"]).strip()
    ]

    texts_to_encode = [chunk["content_embed"] for chunk in filtered_chunks]

    if texts_to_encode:
        vectors = model.encode(texts_to_encode, batch_size=32, show_progress_bar=True)

        points = []
        for i, chunk in enumerate(filtered_chunks):
            chunk.pop("content_embed", None)
            qdrant_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk['chunk_id']))
            points.append(models.PointStruct(id=qdrant_uuid, vector=vectors[i].tolist(), payload=chunk))

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

if __name__ == "__main__":
    INPUT_DIR = os.getenv("DATA_CHUNKED_DIR")

    print("\n=== BẮT ĐẦU PHASE 2: TẠO VECTOR ===")

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(INPUT_DIR, filename)
            print(f"  [+] Đang tạo vector và đẩy DB cho: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            if chunks_data:
                upload_to_qdrant(chunks_data)