---
name: aqua-deployment
description: Deploy LLM models on OCI using AI Quick Actions (AQUA) - single model, multi-model, stacked (LoRA), with GPU shape selection, vLLM configuration, streaming, and tool calling. Triggered when user wants to deploy, update, or manage model deployments.
user-invocable: true
disable-model-invocation: false
---

# AQUA Model Deployment

Use this skill when the user wants to deploy, manage, or configure LLM model deployments on OCI Data Science using AI Quick Actions.

## Deployment Types

| Type | Description |
|---|---|
| **Single Model** | One model per deployment (most common) |
| **Multi-Model** | Multiple LLMs on one instance via LiteLLM routing |
| **Stacked** | Base model + multiple LoRA fine-tuned weights sharing inference |

## Python SDK Usage

### Import

```python
from ads.aqua.modeldeployment import AquaDeploymentApp
deployment_app = AquaDeploymentApp()
```

### Create Single Model Deployment

```python
from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

details = CreateModelDeploymentDetails(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",
    instance_shape="VM.GPU.A10.2",
    display_name="llama-3.1-8b-deployment",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    log_group_id="ocid1.loggroup.oc1.iad.xxx",
    log_id="ocid1.log.oc1.iad.xxx",
    env_var={
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
        "PARAMS": "--max-model-len 4096",
    },
)
deployment = deployment_app.create(create_deployment_details=details)
print(f"Deployment: {deployment.id} | State: {deployment.state}")
```

### Create with Chat Completions Endpoint

```python
details = CreateModelDeploymentDetails(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",
    instance_shape="VM.GPU.A10.2",
    display_name="llama-3.1-8b-chat",
    env_var={
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions",
        "PARAMS": "--max-model-len 4096",
    },
)
```

### Create Multi-Model Deployment

```python
from ads.aqua.common.entities import AquaMultiModelRef

details = CreateModelDeploymentDetails(
    models=[
        AquaMultiModelRef(
            model_id="ocid1.datasciencemodel.oc1.iad.model1",
            model_name="llama-3.1-8b",
            gpu_count=1,
        ),
        AquaMultiModelRef(
            model_id="ocid1.datasciencemodel.oc1.iad.model2",
            model_name="mistral-7b",
            gpu_count=1,
        ),
    ],
    instance_shape="VM.GPU.A10.2",
    display_name="multi-model-deployment",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
)
deployment = deployment_app.create(create_deployment_details=details)
```

### Create Stacked Deployment (Base + LoRA Fine-Tunes)

```python
from ads.aqua.common.entities import AquaMultiModelRef, LoraModuleSpec

details = CreateModelDeploymentDetails(
    models=[
        AquaMultiModelRef(
            model_id="ocid1.datasciencemodel.oc1.iad.base_model",
            model_name="llama-3.1-8b",
            fine_tune_weights=[
                LoraModuleSpec(
                    model_id="ocid1.datasciencemodel.oc1.iad.ft1",
                    model_name="llama-3.1-8b-customer-support",
                ),
                LoraModuleSpec(
                    model_id="ocid1.datasciencemodel.oc1.iad.ft2",
                    model_name="llama-3.1-8b-summarization",
                ),
            ],
        ),
    ],
    instance_shape="VM.GPU.A10.2",
    display_name="stacked-llama-deployment",
    deployment_type="STACKED",
)
deployment = deployment_app.create(create_deployment_details=details)
```

### List Deployments

```python
deployments = deployment_app.list(compartment_id="ocid1.compartment.oc1..xxx")
for d in deployments:
    print(f"{d.display_name} | {d.state} | {d.endpoint}")
```

### Get Deployment Details

```python
deployment = deployment_app.get(model_deployment_id="ocid1.datasciencemodeldeployment.oc1.iad.xxx")
```

### Get Deployment Config (Recommended Shapes)

```python
config = deployment_app.get_deployment_config(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
```

### List Available Shapes

```python
shapes = deployment_app.list_shapes(compartment_id="ocid1.compartment.oc1..xxx")
```

### Shape Recommendation

```python
recommendation = deployment_app.recommend_shape(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
```

## CLI Usage

### Create Deployment
```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "llama-3.1-8b-deployment" \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --log_group_id "ocid1.loggroup.oc1.iad.xxx" \
  --log_id "ocid1.log.oc1.iad.xxx"
```

### Create Multi-Model Deployment
```bash
ads aqua deployment create \
  --models '[{"model_id":"ocid1...model1","model_name":"llama-8b","gpu_count":1},{"model_id":"ocid1...model2","model_name":"mistral-7b","gpu_count":1}]' \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "multi-model"
```

### Create Stacked Deployment
```bash
ads aqua deployment create \
  --models '[{"model_id":"ocid1...base","model_name":"llama-8b","fine_tune_weights":[{"model_id":"ocid1...ft1","model_name":"ft-support"}]}]' \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "stacked-deployment" \
  --deployment_type "STACKED"
```

### List / Get
```bash
ads aqua deployment list --compartment_id "ocid1.compartment.oc1..xxx"
ads aqua deployment get --model_deployment_id "ocid1.datasciencemodeldeployment.oc1.iad.xxx"
```

## Invoking a Deployed Model

### Python SDK (Streaming)

```python
import ads
import oci
import requests

ads.set_auth("resource_principal")
endpoint = "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.iad.xxx"

# Non-streaming
response = requests.post(
    f"{endpoint}/predict",
    json={
        "model": "odsc-llm",
        "prompt": "Write a haiku about clouds",
        "max_tokens": 256,
        "temperature": 0.7,
    },
    auth=oci.auth.signers.get_resource_principals_signer(),
)
print(response.json())
```

### OpenAI-Compatible Client (ADS)

```python
from ads.aqua.client.openai_client import OpenAI

client = OpenAI(
    model_deployment_url="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.iad.xxx",
    auth={"signer": oci.auth.signers.get_resource_principals_signer()},
)
response = client.chat.completions.create(
    model="odsc-llm",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

## GPU Shape Reference

Quick sizing rule: `GPU_memory_GB = num_params_billions × 2` for FP16/BF16, plus ~20% for KV cache.

| Shape | GPUs | GPU Memory | Fits (FP16) |
|---|---|---|---|
| VM.GPU.A10.1 | 1 | 24 GB | ≤ 7B |
| VM.GPU.A10.2 | 2 | 48 GB | ≤ 13B |
| BM.GPU.A10.4 | 4 | 96 GB | ≤ 34B, or 70B quantized |
| BM.GPU.A100-v2.8 | 8 | 640 GB | ≤ 70B |
| BM.GPU.H100.8 | 8 | 640 GB | ≤ 70B (faster) |
| BM.GPU.H200.8 | 8 | 1128 GB | 405B+ |

For the full shape table, per-model recommendations, multi-model GPU count constraints, and quantization options, see `references/shapes.md`.

## vLLM Configuration Parameters

Set via `PARAMS` environment variable or `--params` CLI flag:

| Parameter | Description | Example |
|---|---|---|
| `--max-model-len` | Maximum context length | `4096`, `8192`, `32768` |
| `--gpu-memory-utilization` | Fraction of GPU memory for model | `0.9` (default), `0.95` |
| `--max-num-seqs` | Max concurrent sequences | `256` |
| `--quantization` | Quantization method | `fp8`, `bitsandbytes` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `2`, `4`, `8` |
| `--trust-remote-code` | Allow custom model code from HF | (no value needed) |
| `--enable-auto-tool-choice` | Enable function/tool calling | (no value needed) |
| `--tool-call-parser` | Parser for tool calls | `llama3_json`, `granite`, `hermes` |
| `--limit-mm-per-prompt` | Limit multimodal inputs | `'{"image": 1}'` |
| `--task` | Model task override | `embedding`, `transcribe` |
| `--enforce-eager` | Disable CUDA graphs | (no value needed) |

## Tool Calling / Function Calling

Enable during deployment:
```python
env_var={
    "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions",
    "PARAMS": "--enable-auto-tool-choice --tool-call-parser llama3_json --max-model-len 4096",
}
```

Supported parsers: `llama3_json`, `llama4_json`, `granite`, `hermes`, `mistral`, `jamba`, `pythonic`, `internlm`.

## Advanced Topics

| Topic | Reference |
|---|---|
| Shape recommender CLI + JSON output | `references/shapes.md` → Shape Recommendation Tool section |
| LMCache (KV cache persistence for multi-turn) | `references/lmcache.md` |
| Private endpoints (no public internet) | `references/private-endpoints.md` |
| Batch inferencing (offline Job-based) | `references/batch-inferencing.md` |

## Key Source Files

- `ads/aqua/modeldeployment/deployment.py` — `AquaDeploymentApp` (create, list, get, update)
- `ads/aqua/modeldeployment/entities.py` — `CreateModelDeploymentDetails`, `AquaDeployment`
- `ads/aqua/common/entities.py` — `AquaMultiModelRef`, `LoraModuleSpec`
- `ads/aqua/client/openai_client.py` — OpenAI-compatible client
- `ads/aqua/shaperecommend/recommend.py` — GPU shape recommendation engine
