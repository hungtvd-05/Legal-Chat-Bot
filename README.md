# Final Legal

Hệ thống **GraphRAG** cho bài toán hỏi đáp pháp luật giao thông Việt Nam. Dự án xây dựng toàn bộ pipeline từ thu thập dữ liệu, tách chunk, tạo dữ liệu huấn luyện, fine-tune embedding, hợp nhất mô hình, nạp dữ liệu vào cơ sở dữ liệu vector/graph, đến triển khai API trả lời theo luồng.

## Tổng quan pipeline

1. `crawl_data` – crawl văn bản pháp luật từ `thuvienphapluat.vn`
2. `chunk_and_create_query` – chuẩn hóa dữ liệu, tạo chunk và sinh query huấn luyện
3. `fine_tune_model` – fine-tune embedding model bằng LoRA
4. `merge_and_eval` – merge checkpoint và đánh giá mô hình
5. `upload_data_to_database` – đẩy chunk lên Qdrant và Neo4j
6. `app.py` – API FastAPI để hỏi đáp pháp luật

---

## 1) `crawl_data`

Thư mục `crawl_data` gồm các script crawl dữ liệu pháp luật từ nguồn gốc:

- `crawl.py`: crawl danh sách văn bản theo trang, xử lý đăng nhập, captcha/anti-bot và lưu dữ liệu thô ra JSON
- `crawl_url_amendments.py`: crawl theo danh sách URL có sẵn từ `urls_to_crawl_next.csv`

### Dữ liệu đầu ra

Mỗi văn bản được lưu thành một file JSON riêng, chứa các trường như:

- `source_url`
- `title`
- `metadata`
- `summary_content`
- `mucluc`
- `main_content`
- `amendments`
- `lienquan_hieuluc_list`
- `related_noidung_list`
- `luoc_do_list`

---

## 2) `chunk_and_create_query`

### `chunking.py`

Script này chuyển dữ liệu JSON thô thành các chunk có cấu trúc để phục vụ:

- truy hồi văn bản
- lưu Qdrant
- lưu Neo4j
- tạo dữ liệu huấn luyện

Các loại chunk chính:

- `Document_Meta`
- `Summary` / `Summary_Split`
- `Phần`, `Chương`, `Mục`
- `Điều`, `Khoản`, `Điểm`
- `Preamble`, `Điều_Split`

Mỗi chunk sẽ có các trường quan trọng như:

- `doc_id`
- `chunk_id`
- `parent_chunk_id`
- `content`
- `content_embed`
- `chunk_type`

### `create_query.py`

Script này sinh dữ liệu triplet phục vụ huấn luyện embedding model theo dạng:

- `query`
- `pos`
- `neg`

Nó tạo query từ nội dung chunk bằng LLM, sau đó gắn thêm positive và negative samples để tạo bộ dữ liệu training.

### Đầu ra

- File chunk: `{doc_id}_chunks.json`
- File training: JSONL với format triplet

---

## 3) `fine_tune_model`

### `fine_tune.py`

Script fine-tune embedding model pháp lý với:

- base model: `kietnt0603/nrk-legal-large`
- LoRA trên backbone attention
- loss: `MultipleNegativesRankingLoss`
- evaluator: `InformationRetrievalEvaluator`
- early stopping

### Kết quả

Model sau huấn luyện được lưu trong thư mục:

```text
./nrk-legal-large-traffic-ft-stage2
```

---

## 4) `merge_and_eval`

### `merge_lora.py`

- tải checkpoint LoRA tốt nhất
- merge vào backbone
- lưu model hoàn chỉnh ra thư mục merged

### `eval_model.py`

- tạo bộ evaluation từ dữ liệu chunk và dataset training
- so sánh model gốc và model fine-tuned
- in các metric như `NDCG`, `MRR`, `Accuracy`, `MAP`

### Kết quả

- Model merged: `./nrk-legal-large-traffic-ft-merged-v2`
- Thư mục đánh giá: `./eval_results`

---

## 5) `upload_data_to_database`

### `upload_chunk_to_qdrant.py`

- load model embedding đã fine-tune
- encode `content_embed`
- upload vector và payload lên Qdrant
- collection mặc định: `luat_giao_thong_new_finetune_model`

### `upload_chunk_to_neo4j.py`

- tạo node `Document`, `Chunk`, `SummaryRoot`
- tạo quan hệ:
  - `CHILD_OF`
  - `NEXT_CHUNK`
  - `HAS_SUMMARY`
  - `SUMMARY_PART`
  - `NEXT_PART`

Neo4j được dùng để lấy thêm ngữ cảnh theo cấu trúc văn bản khi trả lời.

---

## 6) `app.py` – API

`app.py` là API FastAPI cuối cùng của hệ thống.

### Luồng xử lý

1. Nhận câu hỏi từ người dùng
2. Mở rộng query bằng LLM
3. Truy hồi dense từ Qdrant
4. Truy hồi sparse bằng BM25
5. Rerank bằng cross-encoder
6. Lấy context liên quan từ Neo4j
7. Gọi LLM sinh câu trả lời cuối cùng
8. Trả kết quả dạng streaming

### Endpoint

```http
POST /api/chat/stream
```

### Chạy API

```bash
python app.py
```

Mặc định service chạy tại:

```text
http://0.0.0.0:8080
```

---

## Cấu hình môi trường

Tạo file `.env` với các biến cần thiết:

```env
MY_USERNAME=your_username
MY_PASSWORD=your_password

DATA_CHUNKED_DIR=/path/to/chunked_data
TRAIN_DATASET=/path/to/training_dataset.jsonl

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

---

## Chạy pipeline theo thứ tự

### 1. Crawl dữ liệu

```bash
python crawl_data/crawl.py
```

hoặc

```bash
python crawl_data/crawl_url_amendments.py
```

### 2. Chunk dữ liệu

```bash
python chunk_and_create_query/chunking.py
```

### 3. Tạo dữ liệu query huấn luyện

```bash
python chunk_and_create_query/create_query.py
```

### 4. Fine-tune embedding model

```bash
python fine_tune_model/fine_tune.py
```

### 5. Merge LoRA

```bash
python merge_and_eval/merge_lora.py
```

### 6. Đánh giá model

```bash
python merge_and_eval/eval_model.py
```

### 7. Upload lên Qdrant

```bash
python upload_data_to_database/upload_chunk_to_qdrant.py
```

### 8. Upload lên Neo4j

```bash
python upload_data_to_database/upload_chunk_to_neo4j.py
```

### 9. Chạy API

```bash
python app.py
```

---

## Yêu cầu hệ thống

- Python 3.10+ khuyến nghị
- Qdrant đang chạy tại `http://localhost:6333`
- Neo4j đang chạy tại `bolt://localhost:7687`
- Có LLM server tương thích OpenAI API cho các bước sinh query và trả lời
- Có quyền truy cập nguồn crawl `thuvienphapluat.vn`

---

## Ghi chú

- Dự án tập trung vào **Luật Giao Thông Việt Nam**
- API trả lời theo kiểu **streaming**
- Hệ thống kết hợp **dense retrieval**, **BM25**, **reranking** và **graph context**

