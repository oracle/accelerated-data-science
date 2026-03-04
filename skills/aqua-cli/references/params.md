# ADS AQUA CLI — Complete Parameter Reference

## `ads aqua model register`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `--model` | Yes | str | HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`) or custom name |
| `--os_path` | Yes | str | Object Storage path: `oci://bucket@namespace/prefix/` |
| `--compartment_id` | No | str | Compartment OCID (defaults to notebook session compartment) |
| `--project_id` | No | str | Project OCID (defaults to notebook session project) |
| `--inference_container` | No | str | Container family name. See container table below. |
| `--inference_container_uri` | No | str | Full container image URI for BYOC deployments |
| `--download_from_hf` | No | bool | `True` to download from HuggingFace Hub |
| `--freeform_tags` | No | JSON str | Freeform tags: `'{"key":"value"}'` |
| `--defined_tags` | No | JSON str | Defined tags |
| `--ignore_patterns` | No | JSON str | Files to exclude: `"['original/*', '*.gitattributes']"` |

### Container Family Options

| `--inference_container` | Use Case |
|---|---|
| `odsc-vllm-serving` | Primary vLLM inference (text, chat, multimodal, embeddings) |
| `odsc-vllm-serving-v1` | vLLM v1 engine |
| `odsc-tgi-serving` | Text Generation Inference (legacy) |
| `odsc-llama-cpp-serving` | llama.cpp for CPU/GGUF models |
| `odsc-tei-serving` | Text Embedding Inference |

---

## `ads aqua deployment create`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `--model_id` | Yes* | str | Model OCID. *Mutually exclusive with `--models` |
| `--models` | Yes* | JSON str | List of model refs for multi/stacked deployment |
| `--instance_shape` | Yes | str | Compute shape (e.g., `VM.GPU.A10.2`) |
| `--display_name` | Yes | str | Deployment display name |
| `--compartment_id` | No | str | Compartment OCID |
| `--project_id` | No | str | Project OCID |
| `--log_group_id` | No | str | Log group OCID (strongly recommended) |
| `--log_id` | No | str | Log OCID (strongly recommended) |
| `--env_var` | No | JSON str | Environment variables: `'{"KEY":"value"}'` |
| `--deployment_type` | No | str | `"STACKED"` for stacked deployments |
| `--container_image_uri` | No | str | Custom container URI (BYOC) |
| `--private_endpoint_id` | No | str | Private endpoint OCID |
| `--subnet_id` | No | str | Subnet OCID for custom egress |
| `--capacity_reservation_ids` | No | JSON str | List of capacity reservation OCIDs |
| `--cmd_var` | No | JSON str | Container command overrides |
| `--instance_count` | No | int | Number of instances (default: 1) |

### Key Environment Variables (`--env_var`)

| Variable | Values | Description |
|---|---|---|
| `MODEL_DEPLOY_PREDICT_ENDPOINT` | `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings` | Inference endpoint mode |
| `PARAMS` | vLLM flags string | Raw vLLM command-line parameters |

### `PARAMS` vLLM Flags

| Flag | Example | Description |
|---|---|---|
| `--max-model-len` | `4096` | Max context window (tokens) |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory allocated |
| `--max-num-seqs` | `256` | Max concurrent request sequences |
| `--quantization` | `fp8`, `bitsandbytes` | Quantization method |
| `--tensor-parallel-size` | `2` | GPUs for tensor parallelism |
| `--trust-remote-code` | _(no value)_ | Allow custom HF model code |
| `--enable-auto-tool-choice` | _(no value)_ | Enable tool/function calling |
| `--tool-call-parser` | `llama3_json` | Tool call response parser |
| `--limit-mm-per-prompt` | `'{"image":1}'` | Limit multimodal inputs per request |
| `--task` | `embedding`, `transcribe` | Override model task |
| `--enforce-eager` | _(no value)_ | Disable CUDA graphs (needed for some models) |
| `--load-format` | `bitsandbytes` | Model loading format |
| `--max-lora-rank` | `64` | Maximum LoRA rank for stacked deployments |

### `--models` JSON Schema

Single model with LoRA weights (stacked):
```json
[{
  "model_id": "ocid1...base",
  "model_name": "llama-3.1-8b",
  "fine_tune_weights": [
    {"model_id": "ocid1...ft1", "model_name": "ft-support"},
    {"model_id": "ocid1...ft2", "model_name": "ft-summary"}
  ]
}]
```

Multi-model (no LoRA):
```json
[
  {"model_id": "ocid1...m1", "model_name": "llama-8b", "gpu_count": 1},
  {"model_id": "ocid1...m2", "model_name": "mistral-7b", "gpu_count": 1}
]
```

---

## `ads aqua fine_tuning create`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `--ft_source_id` | Yes | str | Base model OCID to fine-tune |
| `--ft_name` | Yes | str | Name for the resulting fine-tuned model |
| `--dataset_path` | Yes | str | Object Storage path to JSONL training dataset |
| `--report_path` | Yes | str | Object Storage path for output artifacts |
| `--shape_name` | Yes | str | GPU shape (e.g., `VM.GPU.A10.2`, `BM.GPU.A10.4`) |
| `--replica` | No | int | Number of training nodes (default: 1) |
| `--val_set_size` | No | float | Validation split ratio, e.g., `0.1` for 10% |
| `--ft_parameters` | No | JSON str | Hyperparameters JSON. See table below. |
| `--compartment_id` | No | str | Compartment OCID |
| `--project_id` | No | str | Project OCID |
| `--log_group_id` | No | str | Log group OCID (required for distributed training) |
| `--log_id` | No | str | Log OCID |
| `--subnet_id` | No | str | Subnet OCID (required when `replica > 1`) |
| `--experiment_name` | No | str | Model Version Set name for grouping experiments |
| `--experiment_id` | No | str | Existing Model Version Set OCID |
| `--freeform_tags` | No | JSON str | Tags for the fine-tuned model |

### `--ft_parameters` Hyperparameter Keys

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `epochs` | int | Yes | — | Number of training epochs |
| `learning_rate` | float | Yes | — | Learning rate (e.g., `2e-5`) |
| `batch_size` | int | No | Auto | Micro batch size per GPU |
| `sequence_len` | int | No | Model max | Maximum sequence length |
| `pad_to_sequence_len` | bool | No | `false` | Pad all samples to `sequence_len` |
| `sample_packing` | str/bool | No | `"auto"` | Pack multiple samples per sequence |
| `lora_r` | int | No | `32` | LoRA rank |
| `lora_alpha` | int | No | `16` | LoRA alpha scaling factor |
| `lora_dropout` | float | No | `0.05` | LoRA dropout rate |
| `lora_target_linear` | bool | No | `true` | Target all linear layers |
| `lora_target_modules` | list | No | All linear | Specific module names to target |
| `early_stopping_patience` | int | No | — | Epochs to wait before stopping |
| `early_stopping_threshold` | float | No | — | Min loss improvement to continue |

---

## `ads aqua evaluation create`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `--evaluation_source_id` | Yes | str | Model or model deployment OCID to evaluate |
| `--evaluation_name` | Yes | str | Name for the evaluation |
| `--dataset_path` | Yes | str | Object Storage path to JSONL eval dataset |
| `--report_path` | Yes | str | Object Storage path for HTML/JSON reports |
| `--model_parameters` | Yes | JSON str | Inference parameters. See table below. |
| `--shape_name` | Yes | str | Compute shape for the evaluation job |
| `--block_storage_size` | Yes | int | Block storage in GB (minimum 50) |
| `--compartment_id` | No | str | Compartment OCID |
| `--project_id` | No | str | Project OCID |
| `--log_group_id` | No | str | Log group OCID |
| `--log_id` | No | str | Log OCID |
| `--metrics` | No | JSON str | Metrics array. Default: BERTScore. |
| `--experiment_name` | No | str | Model Version Set name for grouping |
| `--experiment_id` | No | str | Existing Model Version Set OCID |
| `--force_overwrite` | No | bool | Overwrite existing report files |
| `--freeform_tags` | No | JSON str | Tags for the evaluation model |

### `--model_parameters` Keys

| Key | Type | Description |
|---|---|---|
| `max_tokens` | int | Maximum tokens to generate per response |
| `temperature` | float | Sampling temperature (0.0–2.0) |
| `top_p` | float | Nucleus sampling probability |
| `top_k` | int | Top-k sampling |
| `model` | str | For multi/stacked deployments: which model to evaluate |

### `--metrics` Options

| Metric Name | Description |
|---|---|
| `bertscore` | Embedding-based semantic similarity (precision, recall, F1) |
| `rouge` | N-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L) |
| `perplexity` | Model's prediction confidence on the text |
| `text_readability` | Reading level and complexity scores |

Example: `'[{"name": "bertscore"}, {"name": "rouge"}]'`

---

## `ads aqua deployment recommend_shape`

Estimates GPU memory requirements and returns ranked shape recommendations for a model.

**Only supports Safetensor-format decoder-only models from HuggingFace.**

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `--model_id` | Yes | str | — | HuggingFace model name (e.g., `meta-llama/Llama-3.3-70B-Instruct`) or registered model OCID |
| `--generate_table` | No | bool | `True` | `True` = human-readable table; `False` = JSON report |

### Output: Table mode (`generate_table=True`)

Displays a ranked table of shapes with columns: shape name, GPU count, GPU memory, estimated model size, KV cache size, max context length, quantization, recommendation notes.

### Output: JSON mode (`generate_table=False`)

Returns a JSON object with `recommendations` array and `troubleshoot` string. Each recommendation contains:
- `shape_details.name` — shape name
- `shape_details.gpu_specs.ranking.cost` — cost efficiency score (higher = cheaper per GB)
- `shape_details.gpu_specs.ranking.performance` — performance score (higher = faster)
- `configurations[0].deployment_params.max_model_len` — recommended context length
- `configurations[0].deployment_params.quantization` — recommended quantization
- `configurations[0].recommendation` — human-readable fit assessment

### Error Responses

| `troubleshoot` value | Meaning |
|---|---|
| `"Please provide a model in Safetensor format..."` | Model uses GGUF or other format |
| `"Please provide a decoder-only text-generation model..."` | Model is encoder-only or encoder-decoder |

---

## `ads aqua model list` / `get`

| Parameter | Required | Description |
|---|---|---|
| `--compartment_id` | Yes (list) | Compartment to list models from |
| `--model_id` | Yes (get) | Model OCID to retrieve |

## `ads aqua deployment list` / `get`

| Parameter | Required | Description |
|---|---|---|
| `--compartment_id` | Yes (list) | Compartment to list deployments from |
| `--model_deployment_id` | Yes (get) | Deployment OCID to retrieve |

## `ads aqua evaluation list` / `get`

| Parameter | Required | Description |
|---|---|---|
| `--compartment_id` | Yes (list) | Compartment to list evaluations from |
| `--eval_id` | Yes (get) | Evaluation OCID to retrieve |
