import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from transformers import EarlyStoppingCallback
import json
import os
import random
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
#%%
MODEL_NAME   = "kietnt0603/nrk-legal-large"
OUTPUT_DIR   = "./nrk-legal-large-traffic-ft-stage2"
DATASET_PATH = os.getenv("TRAIN_DATASET")
CHUNKS_DIR = os.getenv("DATA_CHUNKED_DIR")
#%%
def build_evaluation_dataset(chunks_dir, training_dataset_path, num_eval_samples=500):
    print("Bắt đầu khởi tạo bộ dữ liệu Evaluation...")
    eval_corpus = {}
    print(f"Đang đọc dữ liệu từ thư mục {chunks_dir} để tạo corpus...")

    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(chunks_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for chunk in data:
                    if "chunk_id" in chunk and "content_embed" in chunk:
                        eval_corpus[chunk["chunk_id"]] = chunk["content_embed"]

    print(f"Đã tải {len(eval_corpus)} tài liệu vào eval_corpus.")
    text_to_chunk_id = {v: k for k, v in eval_corpus.items()}

    all_possible_pairs = []
    print(f"Đang phân tích cú pháp file {training_dataset_path}...")

    with open(training_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                query = item["query"]
                positive_text = item["pos"][0]

                if positive_text in text_to_chunk_id:
                    chunk_id = text_to_chunk_id[positive_text]
                    all_possible_pairs.append({
                        "query": query,
                        "positive_chunk_id": chunk_id
                    })

    print(f"Tìm thấy {len(all_possible_pairs)} cặp câu hỏi-đáp hợp lệ.")

    if len(all_possible_pairs) < num_eval_samples:
        num_eval_samples = len(all_possible_pairs)
        print(f"Tự động giảm số lượng eval xuống: {num_eval_samples}")

    eval_samples = random.sample(all_possible_pairs, num_eval_samples)

    eval_queries = {}
    eval_relevant_docs = {}

    for idx, sample in enumerate(eval_samples):
        query_id = f"q_{idx}"
        eval_queries[query_id] = sample["query"]
        eval_relevant_docs[query_id] = {sample["positive_chunk_id"]}

    print(f"Hoàn thành cấu trúc bộ định giá Eval!")
    return eval_queries, eval_corpus, eval_relevant_docs

eval_queries, eval_corpus, eval_relevant_docs = build_evaluation_dataset(
    chunks_dir=CHUNKS_DIR,
    training_dataset_path=DATASET_PATH,
    num_eval_samples=500
)

raw = load_dataset("json", data_files=DATASET_PATH, split="train")

eval_query_set = set(eval_queries.values())

def flatten(example):
    query = example["query"]
    if query in eval_query_set:
        return {"anchor": None, "positive": None, "negative": None}
    pos = example["pos"]
    neg = example["neg"]
    return {
        "anchor":   query,
        "positive": pos[0] if isinstance(pos, list) else pos,
        "negative": neg[0] if isinstance(neg, list) and len(neg) > 0 else (
                    neg    if isinstance(neg, str)  else ""),
    }

dataset = raw.map(flatten, remove_columns=raw.column_names)
dataset = dataset.filter(lambda x: x["anchor"] is not None and len(x["negative"].strip()) > 10)
print(f"Tổng số lượng dữ liệu thực tế đưa vào huấn luyện: {len(dataset)}")
#%%
print("Đang load model...")
model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": "bfloat16"},
)
model.max_seq_length = 512

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense"
    ],
    bias="none",
    inference_mode=False,
)

backbone = model[0].auto_model
backbone.gradient_checkpointing_enable()
backbone_with_lora = get_peft_model(backbone, peft_config)
backbone_with_lora.print_trainable_parameters()
model[0].auto_model = backbone_with_lora

ir_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    mrr_at_k=[3, 5, 10],
    ndcg_at_k=[3, 5, 10],
    name="legal-traffic-evaluation",
    show_progress_bar=True
)
#%%
training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=540,

    lr_scheduler_type="cosine",

    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,

    metric_for_best_model="eval_legal-traffic-evaluation_cosine_ndcg@10",
    greater_is_better=True,
    load_best_model_at_end=True,

    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    logging_steps=100,
    report_to="none",
)

loss_function = MultipleNegativesRankingLoss(model=model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss_function,
    evaluator=ir_evaluator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
#%%
trainer.train()

LORA_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
model[0].auto_model.save_pretrained(LORA_DIR)

merged_backbone = model[0].auto_model.merge_and_unload()
model[0].auto_model = merged_backbone
model.save_pretrained(OUTPUT_DIR)
print(f"Hoàn thành! Model được lưu tại: {OUTPUT_DIR}")