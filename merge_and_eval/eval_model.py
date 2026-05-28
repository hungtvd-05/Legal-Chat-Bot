from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
import json, os, random
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL   = "kietnt0603/nrk-legal-large"
FINETUNED    = "./nrk-legal-large-traffic-ft-merged-v2"
CHUNKS_DIR   = os.getenv("DATA_CHUNKED_DIR")
DATASET_PATH = os.getenv("TRAIN_DATASET")
OUTPUT_DIR   = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_evaluation_dataset(chunks_dir, training_dataset_path, num_eval_samples=500, seed=42):
    random.seed(seed)
    eval_corpus = {}
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json"):
            with open(os.path.join(chunks_dir, filename), 'r', encoding='utf-8') as f:
                for chunk in json.load(f):
                    if "chunk_id" in chunk and "content_embed" in chunk:
                        eval_corpus[chunk["chunk_id"]] = chunk["content_embed"]

    text_to_chunk_id = {v: k for k, v in eval_corpus.items()}
    all_pairs = []
    with open(training_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pos_text = item["pos"][0]
                if pos_text in text_to_chunk_id:
                    all_pairs.append({
                        "query": item["query"],
                        "positive_chunk_id": text_to_chunk_id[pos_text]
                    })

    num_eval_samples = min(num_eval_samples, len(all_pairs))
    eval_samples = random.sample(all_pairs, num_eval_samples)

    eval_queries, eval_relevant_docs = {}, {}
    for idx, s in enumerate(eval_samples):
        qid = f"q_{idx}"
        eval_queries[qid]       = s["query"]
        eval_relevant_docs[qid] = {s["positive_chunk_id"]}

    return eval_queries, eval_corpus, eval_relevant_docs

print("Building eval dataset...")
eval_queries, eval_corpus, eval_relevant_docs = build_evaluation_dataset(
    CHUNKS_DIR, DATASET_PATH, num_eval_samples=500, seed=42
)
print(f"Queries: {len(eval_queries)}, Corpus: {len(eval_corpus)}")

def make_evaluator(name):
    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        mrr_at_k=[3, 5, 10],
        ndcg_at_k=[3, 5, 10],
        accuracy_at_k=[1, 3, 5, 10],
        name=name,
        show_progress_bar=True,
    )

print("\n" + "="*50)
print("Evaluating FINE-TUNED model...")
ft_model = SentenceTransformer(FINETUNED)
ft_results = make_evaluator("finetuned").__call__(ft_model, output_path=OUTPUT_DIR)
print("FT result keys  :", list(ft_results.keys()))

print("\n" + "="*50)
print("Evaluating BASE model...")
base_model = SentenceTransformer(BASE_MODEL, model_kwargs={"torch_dtype": torch.bfloat16})
base_results = make_evaluator("base").__call__(base_model, output_path=OUTPUT_DIR)
print("Base result keys:", list(base_results.keys()))

metrics = [
    "cosine_ndcg@3", "cosine_ndcg@5", "cosine_ndcg@10",
    "cosine_mrr@3", "cosine_mrr@5", "cosine_mrr@10",
    "cosine_accuracy@1", "cosine_accuracy@3", "cosine_accuracy@5", "cosine_accuracy@10",
    "cosine_map@100",
]

BASE_PREFIX = "base_cosine_"
FT_PREFIX = "finetuned_cosine_"

print("\n" + "=" * 65)
print(f"{'Metric':<25} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
print("-" * 65)
for m in metrics:
    base_key = f"{BASE_PREFIX}{m.split('cosine_')[1]}"
    ft_key = f"{FT_PREFIX}{m.split('cosine_')[1]}"

    base_val = (base_results.get(base_key) or
                base_results.get(f"cosine_{m.split('cosine_')[1]}") or 0)
    ft_val = (ft_results.get(ft_key) or
              ft_results.get(f"cosine_{m.split('cosine_')[1]}") or 0)

    delta = ft_val - base_val
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"{m:<25} {base_val:>10.4f} {ft_val:>12.4f} {arrow}{abs(delta):>8.4f}")

print("=" * 65)