---
name: aqua-finetuning
description: Fine-tune LLM models using LoRA on OCI AI Quick Actions (AQUA). Covers dataset preparation (instruction, conversational, multimodal, tokenized formats), hyperparameter tuning, distributed training, and training metrics. Triggered when user wants to fine-tune or customize a model.
user-invocable: true
disable-model-invocation: false
---

# AQUA Model Fine-Tuning

Use this skill when the user wants to fine-tune LLMs using LoRA on OCI Data Science AI Quick Actions.

## Method: LoRA (Low-Rank Adaptation)

AQUA uses LoRA for parameter-efficient fine-tuning. Default configuration:
```json
{
    "r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}
```
All linear modules are targeted by default.

## Dataset Formats

All datasets must be JSONL format. Every row must be valid JSON with consistent schema.

Four formats are supported — copy the relevant example file from `examples/`:

| Format | File | Use Case |
|---|---|---|
| Instruction | `examples/instruction-format.jsonl` | Completion models; `prompt` + `completion` keys |
| Conversational | `examples/conversational-format.jsonl` | Chat models; `messages` list with `role`/`content` |
| Multimodal (instruction) | `examples/multimodal-format.jsonl` | Mllama vision models; adds `file_name` for image path |
| Multimodal (conversational) | `examples/multimodal-conversational-format.jsonl` | Mllama chat with images |

Note: Instruction format is auto-converted to conversational format for chat models if `chat_template` is available. Tokenized data (`{"input_ids": [...]}`) is also supported — no formatting is applied to it.

## Python SDK Usage

### Import

```python
from ads.aqua.finetuning import AquaFineTuningApp
ft_app = AquaFineTuningApp()
```

### Create Fine-Tuning Job

```python
from ads.aqua.finetuning.entities import CreateFineTuningDetails

details = CreateFineTuningDetails(
    ft_source_id="ocid1.datasciencemodel.oc1.iad.xxx",  # Base model OCID
    ft_name="llama-3.1-8b-customer-support",
    dataset_path="oci://my-bucket@my-namespace/datasets/customer_support.jsonl",
    report_path="oci://my-bucket@my-namespace/ft-output/",
    shape_name="VM.GPU.A10.2",
    replica=1,
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    log_group_id="ocid1.loggroup.oc1.iad.xxx",
    log_id="ocid1.log.oc1.iad.xxx",
    ft_parameters={
        "epochs": 3,
        "learning_rate": 2e-5,
    },
)
ft_job = ft_app.create(create_fine_tuning_details=details)
print(f"Fine-tuning job: {ft_job.id} | State: {ft_job.lifecycle_state}")
```

### With Advanced LoRA Parameters

```python
details = CreateFineTuningDetails(
    ft_source_id="ocid1.datasciencemodel.oc1.iad.xxx",
    ft_name="llama-3.1-8b-custom",
    dataset_path="oci://my-bucket@my-namespace/datasets/train.jsonl",
    report_path="oci://my-bucket@my-namespace/ft-output/",
    shape_name="BM.GPU.A10.4",
    replica=1,
    ft_parameters={
        "epochs": 5,
        "learning_rate": 1e-5,
        "batch_size": 4,
        "sequence_len": 2048,
        "pad_to_sequence_len": True,
        "sample_packing": "auto",
        "lora_r": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_linear": True,
    },
)
```

### With Validation Split

```python
details = CreateFineTuningDetails(
    ft_source_id="ocid1.datasciencemodel.oc1.iad.xxx",
    ft_name="llama-3.1-8b-validated",
    dataset_path="oci://my-bucket@my-namespace/datasets/train.jsonl",
    report_path="oci://my-bucket@my-namespace/ft-output/",
    shape_name="VM.GPU.A10.2",
    replica=1,
    val_set_size=0.1,  # 10% validation split
    ft_parameters={
        "epochs": 3,
        "learning_rate": 2e-5,
    },
)
```

### Get Fine-Tuning Config for a Model

```python
config = ft_app.get_finetuning_config(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
print(config.shape)          # Supported shapes
print(config.configuration)  # Configuration per shape
```

## CLI Usage

### Create Fine-Tuning Job

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

| Parameter | Description | Default |
|---|---|---|
| `epochs` | Number of training epochs | Required |
| `learning_rate` | Learning rate | Required |
| `batch_size` | Micro batch size per GPU | Auto |
| `sequence_len` | Maximum sequence length | Model default |
| `pad_to_sequence_len` | Pad sequences to max length | False |
| `sample_packing` | Pack multiple samples per sequence | `"auto"` |
| `lora_r` | LoRA rank | 32 |
| `lora_alpha` | LoRA alpha scaling | 16 |
| `lora_dropout` | LoRA dropout rate | 0.05 |
| `lora_target_linear` | Target all linear layers | True |
| `lora_target_modules` | Specific modules to target | All linear |
| `early_stopping_patience` | Epochs to wait before early stop | None |
| `early_stopping_threshold` | Min improvement threshold | None |

## Distributed Training

- Use `replica > 1` for multi-node training
- Requires **VCN + Subnet** and **Logging** configuration
- DeepSpeed and FSDP are auto-configured
- Multi-node overhead is significant; only recommended with 5+ replicas
- Single replica with multi-GPU shape (e.g., BM.GPU.A10.4) is preferred when possible

## Training Metrics

At the end of each epoch:
- **Loss**: Should decrease over epochs
- **Accuracy**: Should increase over epochs
- Watch for **overfitting**: validation loss stops decreasing while training loss continues to drop

## Deploying Fine-Tuned Models

Fine-tuned models (V2) are deployed as **stacked deployments** sharing the base model:
```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.fine_tuned_model" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "ft-stacked-deployment"
```

The SDK auto-detects V2 fine-tuned models and creates stacked deployments.

For legacy fine-tuned models, convert first:
```bash
ads aqua model convert_fine_tune --model_id "ocid1.datasciencemodel.oc1.iad.legacy_ft"
```

## Key Source Files

- `ads/aqua/finetuning/finetuning.py` — `AquaFineTuningApp` (create, get config)
- `ads/aqua/finetuning/entities.py` — `CreateFineTuningDetails`, `AquaFineTuningParams`
- `ads/aqua/finetuning/constants.py` — Fine-tuning metadata keys, restricted params
