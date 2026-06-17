---
name: aqua-finetuning
description: "Fine-tune, train, adapt, or retrain LLM models using LoRA (PEFT) on OCI AI Quick Actions (AQUA). Covers dataset preparation (instruction, conversational, multimodal, tokenized JSONL formats), hyperparameter tuning, distributed multi-node training, validation splits, and training metrics monitoring. Triggered when user wants to fine-tune, customize, adapt, retrain, or transfer-learn a model on OCI Data Science."
user-invocable: true
disable-model-invocation: false
---

# AQUA Model Fine-Tuning

Fine-tune LLMs using LoRA (PEFT) on OCI Data Science AI Quick Actions. Use when the user wants to fine-tune, train, adapt, retrain, or transfer-learn a model on AQUA.

## End-to-End Workflow

1. **Prepare dataset** → validate JSONL format (see Dataset Formats below)
2. **Check model config** → `ft_app.get_finetuning_config(model_id=...)` to confirm supported shapes
3. **Create fine-tuning job** → submit via SDK or CLI
4. **Monitor status** → poll `ft_app.get(ft_job.id).lifecycle_state` until `SUCCEEDED` or `FAILED`
5. **Review metrics** → check loss/accuracy trends in the report path output
6. **Deploy fine-tuned model** → create stacked deployment

## Dataset Formats

All datasets must be JSONL. Every row must be valid JSON with consistent schema.

| Format | File | Keys |
|---|---|---|
| Instruction | `examples/instruction-format.jsonl` | `{"prompt": "...", "completion": "..."}` |
| Conversational | `examples/conversational-format.jsonl` | `{"messages": [{"role": "...", "content": "..."}]}` |
| Multimodal (instruction) | `examples/multimodal-format.jsonl` | Adds `"file_name"` for image path |
| Multimodal (conversational) | `examples/multimodal-conversational-format.jsonl` | `{"conversations": [...], "file_name": "..."}` |

Instruction format is auto-converted to conversational for chat models when `chat_template` is available. Tokenized data (`{"input_ids": [...]}`) is also supported — no formatting is applied.

## LoRA Defaults

```json
{"r": 32, "lora_alpha": 16, "lora_dropout": 0.05}
```
All linear modules are targeted by default.

## Python SDK Usage

```python
from ads.aqua.finetuning import AquaFineTuningApp
from ads.aqua.finetuning.entities import CreateFineTuningDetails

ft_app = AquaFineTuningApp()

# Step 1: Check supported shapes for the base model
config = ft_app.get_finetuning_config(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
print(config.shape, config.configuration)

# Step 2: Create fine-tuning job
details = CreateFineTuningDetails(
    ft_source_id="ocid1.datasciencemodel.oc1.iad.xxx",
    ft_name="llama-3.1-8b-customer-support",
    dataset_path="oci://my-bucket@my-namespace/datasets/customer_support.jsonl",
    report_path="oci://my-bucket@my-namespace/ft-output/",
    shape_name="VM.GPU.A10.2",
    replica=1,
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    log_group_id="ocid1.loggroup.oc1.iad.xxx",
    log_id="ocid1.log.oc1.iad.xxx",
    ft_parameters={"epochs": 3, "learning_rate": 2e-5},
)
ft_job = ft_app.create(create_fine_tuning_details=details)

# Step 3: Monitor until complete
import time
while ft_job.lifecycle_state not in ("SUCCEEDED", "FAILED"):
    time.sleep(60)
    ft_job = ft_app.get(ft_job.id)
    print(f"State: {ft_job.lifecycle_state}")
```

**Variations** — add these parameters to `CreateFineTuningDetails` as needed:

| Variation | Additional parameters |
|---|---|
| Advanced LoRA | `ft_parameters={..., "lora_r": 64, "lora_alpha": 32, "lora_dropout": 0.1, "lora_target_linear": True, "batch_size": 4, "sequence_len": 2048, "sample_packing": "auto"}` |
| Validation split | `val_set_size=0.1` (10% held out for validation loss tracking) |
| Multi-GPU single node | `shape_name="BM.GPU.A10.4"` (preferred over multi-node when possible) |
| Multi-node distributed | `replica=5` (requires VCN + Subnet + Logging; DeepSpeed/FSDP auto-configured) |

## CLI Usage

```bash
ads aqua fine_tuning create \
  --ft_source_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --ft_name "llama-3.1-8b-customer-support" \
  --dataset_path "oci://my-bucket@my-namespace/datasets/train.jsonl" \
  --report_path "oci://my-bucket@my-namespace/ft-output/" \
  --shape_name "VM.GPU.A10.2" \
  --replica 1 \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --log_group_id "ocid1.loggroup.oc1.iad.xxx" \
  --log_id "ocid1.log.oc1.iad.xxx" \
  --ft_parameters '{"epochs": 3, "learning_rate": 0.00002}'
```

## Hyperparameters Reference

| Parameter | Default | Notes |
|---|---|---|
| `epochs` | Required | Number of training epochs |
| `learning_rate` | Required | Typical: 1e-5 to 5e-5 |
| `batch_size` | Auto | Micro batch size per GPU |
| `sequence_len` | Model default | Maximum sequence length |
| `pad_to_sequence_len` | False | Pad sequences to max length |
| `sample_packing` | `"auto"` | Pack multiple samples per sequence |
| `lora_r` | 32 | LoRA rank — higher = more capacity, more VRAM |
| `lora_alpha` | 16 | LoRA alpha scaling factor |
| `lora_dropout` | 0.05 | LoRA dropout rate |
| `lora_target_linear` | True | Target all linear layers |
| `lora_target_modules` | All linear | Specific modules to target |
| `early_stopping_patience` | None | Epochs to wait before early stop |
| `early_stopping_threshold` | None | Min improvement threshold |

## Deploying Fine-Tuned Models

Fine-tuned models (V2) deploy as stacked deployments sharing the base model:
```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.fine_tuned_model" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "ft-stacked-deployment"
```

The SDK auto-detects V2 fine-tuned models and creates stacked deployments. For legacy models, convert first:
```bash
ads aqua model convert_fine_tune --model_id "ocid1.datasciencemodel.oc1.iad.legacy_ft"
```

## Key Source Files

- `ads/aqua/finetuning/finetuning.py` — `AquaFineTuningApp` (create, get config)
- `ads/aqua/finetuning/entities.py` — `CreateFineTuningDetails`, `AquaFineTuningParams`
- `ads/aqua/finetuning/constants.py` — Fine-tuning metadata keys, restricted params
