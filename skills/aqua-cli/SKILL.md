---
name: aqua-cli
description: Complete CLI reference for the ADS AQUA command-line interface (ads aqua). Covers all model, deployment, evaluation, and fine-tuning commands with full parameter documentation. Triggered when user asks about CLI commands, wants to run AQUA operations from terminal, or needs command syntax.
user-invocable: true
disable-model-invocation: false
---

# ADS AQUA CLI Reference

The `ads aqua` CLI provides command-line access to all AI Quick Actions operations. It uses Python Fire under the hood.

## Installation

```bash
pip install oracle-ads[aqua]
# OR for development
pip install -e ".[aqua]"
```

## Authentication Setup

```bash
# For OCI Notebook Sessions (Resource Principal)
# No setup needed - automatic

# For local development with security token
export OCI_IAM_TYPE="security_token"
export OCI_CONFIG_PROFILE=<your-profile>

# For local development with API key
export OCI_IAM_TYPE="api_key"
export OCI_CONFIG_PROFILE=DEFAULT
```

---

## Model Commands

### List Models

```bash
ads aqua model list \
  --compartment_id "ocid1.compartment.oc1..xxx"
```

### Get Model Details

```bash
ads aqua model get \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx"
```

### Register Model from HuggingFace

```bash
ads aqua model register \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --os_path "oci://my-bucket@my-namespace/models/llama-3.1-8b/" \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --inference_container "odsc-vllm-serving" \
  --download_from_hf True
```

Full parameter reference: `references/params.md`

### Register from Object Storage

```bash
ads aqua model register \
  --model "my-custom-model" \
  --os_path "oci://my-bucket@my-namespace/models/custom-model/" \
  --inference_container "odsc-vllm-serving"
```

### Register GGUF Model

```bash
ads aqua model register \
  --model "TheBloke/Llama-2-7B-Chat-GGUF" \
  --os_path "oci://my-bucket@my-namespace/models/llama2-gguf/" \
  --inference_container "odsc-llama-cpp-serving" \
  --download_from_hf True
```

### Register with BYOC (Bring Your Own Container)

```bash
ads aqua model register \
  --model "my-custom-model" \
  --os_path "oci://my-bucket@my-namespace/models/custom/" \
  --inference_container_uri "<region>.ocir.io/<namespace>/<repo>:<tag>"
```

### Convert Legacy Fine-Tuned Model

```bash
ads aqua model convert_fine_tune \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx"
```

---

## Deployment Commands

### Create Single Model Deployment

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "my-deployment" \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --log_group_id "ocid1.loggroup.oc1.iad.xxx" \
  --log_id "ocid1.log.oc1.iad.xxx"
```

Full parameter reference: `references/params.md`

### Create with Custom vLLM Parameters

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "my-deployment" \
  --env_var '{"MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions", "PARAMS": "--max-model-len 8192 --gpu-memory-utilization 0.95"}'
```

### Create Multi-Model Deployment

```bash
ads aqua deployment create \
  --models '[
    {"model_id": "ocid1...model1", "model_name": "llama-8b", "gpu_count": 1},
    {"model_id": "ocid1...model2", "model_name": "mistral-7b", "gpu_count": 1}
  ]' \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "multi-model-deployment"
```

### Create Stacked Deployment

```bash
ads aqua deployment create \
  --models '[
    {
      "model_id": "ocid1...base_model",
      "model_name": "llama-3.1-8b",
      "fine_tune_weights": [
        {"model_id": "ocid1...ft1", "model_name": "ft-customer-support"},
        {"model_id": "ocid1...ft2", "model_name": "ft-summarization"}
      ]
    }
  ]' \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "stacked-deployment" \
  --deployment_type "STACKED"
```

### Deploy with Tool Calling

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "tool-calling-deployment" \
  --env_var '{"MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions", "PARAMS": "--enable-auto-tool-choice --tool-call-parser llama3_json --max-model-len 4096"}'
```

### Deploy GGUF Model on CPU

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.Standard.A1.Flex" \
  --display_name "cpu-gguf-deployment" \
  --env_var '{"PARAMS": "--quantization Q4_0"}'
```

### Shape Recommendation

```bash
# Table output (human-friendly default)
ads aqua deployment recommend_shape \
  --model_id "meta-llama/Llama-3.3-70B-Instruct"

# By model OCID
ads aqua deployment recommend_shape \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx"

# JSON output (programmatic use)
ads aqua deployment recommend_shape \
  --model_id "meta-llama/Llama-3.3-70B-Instruct" \
  --generate_table False
```

Full parameter reference: `references/params.md`

### List Deployments

```bash
ads aqua deployment list \
  --compartment_id "ocid1.compartment.oc1..xxx"
```

### Get Deployment Details

```bash
ads aqua deployment get \
  --model_deployment_id "ocid1.datasciencemodeldeployment.oc1.iad.xxx"
```

---

## Fine-Tuning Commands

### Create Fine-Tuning Job

```bash
ads aqua fine_tuning create \
  --ft_source_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --ft_name "llama-3.1-8b-custom" \
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

Full parameter reference: `references/params.md`

### Advanced Hyperparameters

```bash
ads aqua fine_tuning create \
  --ft_source_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --ft_name "llama-custom-advanced" \
  --dataset_path "oci://bucket@ns/train.jsonl" \
  --report_path "oci://bucket@ns/output/" \
  --shape_name "BM.GPU.A10.4" \
  --replica 1 \
  --ft_parameters '{
    "epochs": 5,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "sequence_len": 2048,
    "pad_to_sequence_len": true,
    "sample_packing": "auto",
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_linear": true
  }'
```

---

## Evaluation Commands

### Create Evaluation

```bash
ads aqua evaluation create \
  --evaluation_source_id "ocid1.datasciencemodeldeployment.oc1.iad.xxx" \
  --evaluation_name "llama-eval-bertscore" \
  --dataset_path "oci://my-bucket@my-namespace/datasets/eval.jsonl" \
  --report_path "oci://my-bucket@my-namespace/eval-reports/" \
  --model_parameters '{"max_tokens": 500, "temperature": 0.7}' \
  --shape_name "VM.Standard.E4.Flex" \
  --block_storage_size 50 \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --metrics '[{"name": "bertscore"}, {"name": "rouge"}]'
```

Full parameter reference: `references/params.md`

### Evaluate Specific Model in Multi/Stacked Deployment

```bash
ads aqua evaluation create \
  --evaluation_source_id "ocid1.datasciencemodeldeployment.oc1.iad.xxx" \
  --evaluation_name "stacked-ft1-eval" \
  --dataset_path "oci://bucket@ns/eval.jsonl" \
  --report_path "oci://bucket@ns/eval-reports/" \
  --model_parameters '{"max_tokens": 500, "temperature": 0.7, "model": "ft-customer-support"}' \
  --shape_name "VM.Standard.E4.Flex" \
  --block_storage_size 50 \
  --metrics '[{"name": "bertscore"}]'
```

### List Evaluations

```bash
ads aqua evaluation list \
  --compartment_id "ocid1.compartment.oc1..xxx"
```

### Get Evaluation Details

```bash
ads aqua evaluation get \
  --eval_id "ocid1.datasciencemodel.oc1.iad.xxx"
```

---

## Policy Verification

```bash
ads aqua verify_policies
```

---

## Watching Logs

```bash
# Model deployment logs
ads opctl watch <model_deployment_ocid> --auth resource_principal

# Job run logs (fine-tuning / evaluation)
ads opctl watch <job_run_ocid> --auth resource_principal
```

## Key Source Files

- `ads/aqua/cli.py` — `AquaCommand` entry point (model, deployment, evaluation, fine_tuning)
- `ads/aqua/app.py` — `CLIBuilderMixin` for CLI parameter handling
