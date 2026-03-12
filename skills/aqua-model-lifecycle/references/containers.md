# AQUA Inference Container Versions

Current versions of inference containers supported in AI Quick Actions.

## Container Version Table

| Server | Version | Supported Formats | Supported Shapes | Notes |
|---|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm/releases/tag/v0.11.0) | **0.11.0** | safe-tensors | A10, A100, H100, H200 | Primary recommendation |
| [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference/releases/tag/v3.2.1) | **3.2.1** | safe-tensors | A10, A100, H100 | Deprecated — migrate to vLLM |
| [Llama-cpp](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.7) | **0.3.7** | gguf | Ampere ARM | CPU/ARM deployment only |

## Container Family Names (for `inference_container` parameter)

| Family | Version Used | Use Case |
|---|---|---|
| `odsc-vllm-serving` | Latest vLLM (0.11.0) | Text generation, chat, embeddings, multimodal, ASR |
| `odsc-vllm-serving-v1` | vLLM v1 engine | vLLM v1 engine (experimental) |
| `odsc-tgi-serving` | TGI 3.2.1 | Legacy — use vLLM instead |
| `odsc-llama-cpp-serving` | Llama-cpp 0.3.7 | GGUF models on CPU/ARM shapes |
| `odsc-tei-serving` | Latest TEI | Text embedding models only |

## vLLM Supported Models (v0.11.0)

Full list: https://docs.vllm.ai/en/v0.11.0/models/supported_models.html

Key architecture families supported:
- LLaMA / LLaMA 2 / LLaMA 3 (text + vision variants)
- Mistral / Mixtral
- Falcon
- Phi / Phi-3
- Gemma / Gemma 2
- Qwen / Qwen2
- Granite (IBM)
- GPT-J, GPT-NeoX
- Embedding models (via `--task embedding`)
- ASR models: Whisper, Qwen2-Audio, Granite-Speech (via `--task transcribe`)
- Vision-language models via `--limit-mm-per-prompt`

## TGI Supported Models (v3.2.1)

Full list: https://github.com/huggingface/text-generation-inference/blob/v3.2.1/docs/source/supported_models.md

> **Note:** TGI is deprecated in AQUA. New models should use `odsc-vllm-serving`. For migrating existing TGI-registered models, see `tgi-migration.md`.

## Llama-cpp Supported Models (v0.3.7)

Supports GGUF-quantized versions of any model compatible with llama.cpp. GGUF files must be present in the model artifact. Requires Ampere ARM shapes (e.g., `VM.Standard.A1.Flex`).

## Checking the Container Version in a Registration

The container version used is stored in the model's `custom_metadata_list` under the `deployment-container` key. To inspect:

```python
from oci.data_science import DataScienceClient
from ads import set_auth

set_auth(auth='resource_principal')
client = DataScienceClient(config={}, signer=...)

model = client.get_model(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
for meta in model.data.custom_metadata_list:
    if meta.key == "deployment-container":
        print(meta.value)   # e.g., "odsc-vllm-serving"
```

## Using a Specific Container Version (dsmc URI)

For batch jobs or custom container use, reference specific versions directly:

```python
container_image = "dsmc://odsc-vllm-serving:0.6.2"   # pin to specific version
```

The `dsmc://` scheme is resolved by OCI Data Science to the container registry image.
