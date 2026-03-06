---
name: aqua-model-lifecycle
description: Register, list, get, and manage LLM models in OCI AI Quick Actions (AQUA) using the ADS SDK. Triggered when user wants to import models from HuggingFace or Object Storage, browse available models, or manage model catalog entries.
user-invocable: true
disable-model-invocation: false
---

# AQUA Model Lifecycle Management

Use this skill when the user wants to register, list, browse, or manage LLM models in OCI Data Science AI Quick Actions (AQUA).

## Prerequisites

1. **OCI Policies** must be in place (see aqua-troubleshooting skill for policy details).
2. **Object Storage bucket** with versioning enabled.
3. **Authentication** configured (Resource Principal in notebook, or API Key locally).

```python
import ads
ads.set_auth("resource_principal")  # In OCI notebook sessions
# OR
ads.set_auth("api_key")            # Local development
```

## Python SDK Usage

### Import

```python
from ads.aqua.model import AquaModelApp
model_app = AquaModelApp()
```

### List Models

```python
# List all AQUA models in a compartment
models = model_app.list(compartment_id="ocid1.compartment.oc1..xxx")
for m in models:
    print(f"{m.display_name} | {m.id} | {m.lifecycle_state}")
```

### Get Model Details

```python
model = model_app.get(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
print(model.display_name)
print(model.lifecycle_state)
```

### Register a Model from HuggingFace

```python
from ads.aqua.model.entities import ImportModelDetails

details = ImportModelDetails(
    model="meta-llama/Llama-3.1-8B-Instruct",
    os_path="oci://my-bucket@my-namespace/models/llama-3.1-8b/",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    inference_container="odsc-vllm-serving",
    download_from_hf=True,
)
model = model_app.register(import_model_details=details)
print(f"Registered: {model.id}")
```

### Register a Model from Object Storage

If model artifacts are already uploaded to Object Storage:

```python
details = ImportModelDetails(
    model="my-custom-model",
    os_path="oci://my-bucket@my-namespace/models/custom-model/",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    inference_container="odsc-vllm-serving",
)
model = model_app.register(import_model_details=details)
```

### Register with Freeform Tags

```python
details = ImportModelDetails(
    model="meta-llama/Llama-3.1-8B-Instruct",
    os_path="oci://my-bucket@my-namespace/models/llama-3.1-8b/",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    inference_container="odsc-vllm-serving",
    download_from_hf=True,
    freeform_tags={"team": "ml-platform", "env": "dev"},
)
```

## CLI Usage

### List Models
```bash
ads aqua model list --compartment_id ocid1.compartment.oc1..xxx
```

### Get Model
```bash
ads aqua model get --model_id ocid1.datasciencemodel.oc1.iad.xxx
```

### Register from HuggingFace
```bash
ads aqua model register \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --os_path "oci://my-bucket@my-namespace/models/llama-3.1-8b/" \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --inference_container "odsc-vllm-serving" \
  --download_from_hf True
```

### Register from Object Storage
```bash
ads aqua model register \
  --model "my-custom-model" \
  --os_path "oci://my-bucket@my-namespace/models/custom-model/" \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --inference_container "odsc-vllm-serving"
```

### Register GGUF Model (for CPU deployment)
```bash
ads aqua model register \
  --model "TheBloke/Llama-2-7B-Chat-GGUF" \
  --os_path "oci://my-bucket@my-namespace/models/llama2-gguf/" \
  --inference_container "odsc-llama-cpp-serving" \
  --download_from_hf True
```

## Supported Inference Containers

| Container Family | Use Case |
|---|---|
| `odsc-vllm-serving` | Primary vLLM inference (text generation, chat, multimodal) |
| `odsc-vllm-serving-v1` | vLLM v1 inference engine |
| `odsc-tgi-serving` | Text Generation Inference (deprecated, migrate to vLLM) |
| `odsc-llama-cpp-serving` | llama.cpp for CPU/ARM deployment with GGUF models |
| `odsc-tei-serving` | Text Embedding Inference |

## Model Artifact Filtering

Exclude unnecessary files during HuggingFace download using wildcards:
```bash
ads aqua model register \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --os_path "oci://my-bucket@my-namespace/models/llama-3.1-8b/" \
  --inference_container "odsc-vllm-serving" \
  --download_from_hf True \
  --ignore_patterns "['original/*', '*.gitattributes']"
```

## Gated Models (HuggingFace)

For gated repositories (e.g., Llama, Mistral), authenticate first:
```bash
export HF_TOKEN=<your_hf_read_token>
# OR
huggingface-cli login
```

## Advanced Topics

| Topic | Reference |
|---|---|
| Current container versions (vLLM 0.11.0, TGI 3.2.1, Llama-cpp 0.3.7) | `references/containers.md` |
| Migrating TGI-registered models to vLLM (affected: gemma, codegemma, falcon-40b) | `references/tgi-migration.md` |

## Key Source Files

- `ads/aqua/model/model.py` — `AquaModelApp` class (register, list, get, delete)
- `ads/aqua/model/entities.py` — `AquaModel`, `AquaModelSummary`, `ImportModelDetails`
- `ads/aqua/model/constants.py` — Model metadata fields, model types
- `ads/aqua/cli.py` — CLI entry point (`ads aqua model ...`)
