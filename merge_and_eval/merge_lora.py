from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType
import torch, os, safetensors.torch as st

BASE_MODEL = "kietnt0603/nrk-legal-large"
BEST_CKPT = os.path.abspath("./nrk-legal-large-traffic-ft-stage2/checkpoint-3612")
MERGED_DIR = os.path.abspath("./nrk-legal-large-traffic-ft-merged-v2")

base_model = SentenceTransformer(BASE_MODEL, model_kwargs={"torch_dtype": torch.bfloat16})
backbone   = base_model[0].auto_model

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["attention.self.query", "attention.self.key",
                    "attention.self.value", "attention.output.dense"],
    bias="none", inference_mode=True,
)
backbone_lora = get_peft_model(backbone, peft_config)

ckpt_file = os.path.join(BEST_CKPT, "model.safetensors")
saved_state = st.load_file(ckpt_file)
print(f"Keys trong checkpoint: {len(saved_state)}")
print(f"Sample key: {list(saved_state.keys())[0]}")

remapped = {"base_model.model." + k: v for k, v in saved_state.items()}
missing, unexpected = backbone_lora.load_state_dict(remapped, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

merged = backbone_lora.merge_and_unload()
base_model[0].auto_model = merged

base_ref = SentenceTransformer(BASE_MODEL, model_kwargs={"torch_dtype": torch.bfloat16})
for name, param in merged.named_parameters():
    if "attention.self.query.weight" in name:
        ref = dict(base_ref[0].auto_model.named_parameters())[name]
        diff = (param.float() - ref.float()).abs().mean()
        print(f"abs diff: {diff:.6f}  ← phải > 0")
        break

base_model.save_pretrained(MERGED_DIR)
print(f"Saved best model to {MERGED_DIR}")
