# GPU Shape & Model Compatibility Reference

## OCI GPU Shapes

| Shape | GPUs | Total GPU Memory | GPU Type | Use Case |
|---|---|---|---|---|
| VM.GPU.A10.1 | 1 | 24 GB | NVIDIA A10 | Models up to 7B (FP16) |
| VM.GPU.A10.2 | 2 | 48 GB | NVIDIA A10 | Models up to 13B (FP16), 7B with large context |
| BM.GPU.A10.4 | 4 | 96 GB | NVIDIA A10 | Models up to 34B (FP16), 70B quantized (FP8) |
| BM.GPU4.8 | 8 | 320 GB | NVIDIA A100 40GB | Models up to 70B (FP16) |
| BM.GPU.A100-v2.8 | 8 | 640 GB | NVIDIA A100 80GB | Models up to 405B (FP16), 70B with very large context |
| BM.GPU.H100.8 | 8 | 640 GB | NVIDIA H100 80GB | High-throughput 70B+ models |
| BM.GPU.H200.8 | 8 | 1128 GB | NVIDIA H200 141GB | Very large models, 405B+ |
| BM.GPU.L40S-NC.4 | 4 | 192 GB | NVIDIA L40S | Cost-effective mid-range |
| BM.GPU.B4.8 | 8 | 320 GB | NVIDIA B200 | Next-gen Blackwell |
| BM.GPU.MI300X.8 | 8 | 1536 GB | AMD MI300X | Very large models, AMD |
| VM.GPU2.1 | 1 | 16 GB | NVIDIA P100 | Legacy, small models only |
| VM.GPU3.1 | 1 | 16 GB | NVIDIA V100 | Legacy |
| VM.GPU3.2 | 2 | 32 GB | NVIDIA V100 | Legacy |
| VM.GPU3.4 | 4 | 64 GB | NVIDIA V100 | Legacy |

## Memory Estimation Guide

| Precision | Bytes per Parameter | 7B Model | 13B Model | 34B Model | 70B Model | 405B Model |
|---|---|---|---|---|---|---|
| FP32 | 4 | 28 GB | 52 GB | 136 GB | 280 GB | 1620 GB |
| FP16 / BF16 | 2 | 14 GB | 26 GB | 68 GB | 140 GB | 810 GB |
| FP8 / INT8 | 1 | 7 GB | 13 GB | 34 GB | 70 GB | 405 GB |
| INT4 | 0.5 | 3.5 GB | 6.5 GB | 17 GB | 35 GB | 202 GB |

**Rule of thumb**: Add ~20% to model weight memory for KV cache at moderate context lengths.

## Recommended Shape by Model Size (FP16)

| Model Size | Recommended Shape | Notes |
|---|---|---|
| ≤ 7B | VM.GPU.A10.1 | 24 GB fits 7B with context |
| 7B–13B | VM.GPU.A10.2 | 48 GB fits 13B comfortably |
| 13B–34B | BM.GPU.A10.4 | 96 GB; can run 70B quantized |
| 34B–70B | BM.GPU4.8 or BM.GPU.A100-v2.8 | 320–640 GB |
| 70B+ | BM.GPU.A100-v2.8 or BM.GPU.H100.8 | 640 GB for full FP16 |
| 405B | BM.GPU.H200.8 | 1128 GB |

## Service-Managed Model Recommendations

Models listed here are Oracle-tested with confirmed working shapes and parameters.

| Model | Shape | GPU Count | Key vLLM Params |
|---|---|---|---|
| CodeLlama-7b-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 4096` |
| CodeLlama-13b-Instruct | VM.GPU.A10.2 | 2 | `--max-model-len 4096` |
| CodeLlama-34b-Instruct | BM.GPU.A10.4 | 4 | `--max-model-len 4096` |
| Granite-3.0-2B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 4096` |
| Granite-3.0-8B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 4096` |
| Llama-3-8B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 8192` |
| Llama-3-70B-Instruct | BM.GPU.A100-v2.8 | 8 | `--max-model-len 8192` |
| Llama-3.1-8B-Instruct | VM.GPU.A10.2 | 2 | `--max-model-len 32768` |
| Llama-3.1-70B-Instruct | BM.GPU.A100-v2.8 | 8 | `--max-model-len 32768` |
| Llama-3.1-405B-Instruct | BM.GPU.H200.8 | 8 | `--max-model-len 16384` |
| Llama-3.2-1B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 8192` |
| Llama-3.2-3B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 8192` |
| Llama-3.2-11B-Vision-Instruct | VM.GPU.A10.2 | 2 | `--max-model-len 8192 --limit-mm-per-prompt '{"image":1}'` |
| Llama-3.2-90B-Vision-Instruct | BM.GPU.A100-v2.8 | 8 | `--max-model-len 8192 --limit-mm-per-prompt '{"image":1}'` |
| Mistral-7B-Instruct-v0.3 | VM.GPU.A10.1 | 1 | `--max-model-len 4096` |
| Mixtral-8x7B-Instruct-v0.1 | BM.GPU.A10.4 | 4 | `--max-model-len 4096` |
| Phi-3-mini-4k-instruct | VM.GPU.A10.1 | 1 | `--max-model-len 4096 --trust-remote-code` |
| Phi-3-medium-128k-instruct | VM.GPU.A10.2 | 2 | `--max-model-len 8192 --trust-remote-code` |
| Phi-3-vision-128k-instruct | VM.GPU.A10.2 | 2 | `--max-model-len 4096 --trust-remote-code --limit-mm-per-prompt '{"image":1}'` |
| Falcon-7B-Instruct | VM.GPU.A10.1 | 1 | `--max-model-len 2048` |
| Falcon-40B-Instruct | BM.GPU.A10.4 | 4 | `--max-model-len 2048` |
| GPT-J-6B | VM.GPU.A10.1 | 1 | `--max-model-len 2048` |

## Multi-Model Deployment: GPU Count Constraints

- GPU counts per model **must be powers of 2**: 1, 2, 4, 8
- All models in a multi-model deployment **must use the same shape**
- Total GPU count across all models must not exceed the shape's GPU count

| Shape | Total GPUs | Example Configurations |
|---|---|---|
| VM.GPU.A10.2 | 2 | 2×(1 GPU each), 1×(2 GPU) |
| BM.GPU.A10.4 | 4 | 4×(1 GPU each), 2×(2 GPU each), 1×(4 GPU) |
| BM.GPU.A100-v2.8 | 8 | 8×(1 GPU), 4×(2 GPU), 2×(4 GPU), 1×(8 GPU) |

## Quantization Options

| Method | Flag | Precision | Memory Savings | Quality Impact |
|---|---|---|---|---|
| FP8 | `--quantization fp8` | 8-bit | ~50% vs FP16 | Minimal |
| bitsandbytes (4-bit) | `--quantization bitsandbytes --load-format bitsandbytes` | 4-bit | ~75% vs FP16 | Moderate |
| GGUF Q2_K | (via llama.cpp container) | ~2-bit | Extreme | High |
| GGUF Q4_0 | (via llama.cpp container) | ~4-bit | Large | Low–Moderate |
| GGUF Q5_0 | (via llama.cpp container) | ~5-bit | Moderate | Low |
| GGUF Q8_0 | (via llama.cpp container) | ~8-bit | Small | Minimal |

GGUF quantization requires the `odsc-llama-cpp-serving` container and models registered with GGUF artifacts.

## Shape Recommendation Tool

The `recommend_shape` command automatically estimates GPU memory requirements and returns ranked shape recommendations for any Safetensor-format HuggingFace model.

### What It Analyzes

- **Model weight memory** — estimated from the model's `config.json` (parameter count × bytes per precision)
- **KV cache memory** — estimated at various context lengths
- **Total fit** — checks against each shape's GPU memory, applies quantization if needed to fit

### CLI Usage

```bash
# By HuggingFace model name (table output — best for humans)
ads aqua deployment recommend_shape --model_id meta-llama/Llama-3.3-70B-Instruct

# By registered model OCID
ads aqua deployment recommend_shape --model_id ocid1.datasciencemodel.oc1.iad.xxx

# JSON output (best for programmatic use)
ads aqua deployment recommend_shape \
  --model_id meta-llama/Llama-3.3-70B-Instruct \
  --generate_table False
```

### Python SDK Usage

```python
from ads.aqua.modeldeployment import AquaDeploymentApp

# Returns JSON report
result = AquaDeploymentApp().recommend_shape(
    model_id="meta-llama/Llama-3.3-70B-Instruct"
)
for rec in result["recommendations"]:
    shape = rec["shape_details"]["name"]
    config = rec["configurations"][0]
    print(f"{shape}: {config['recommendation']}")
```

### Output Fields (JSON mode, `generate_table=False`)

```json
{
  "recommendations": [
    {
      "shape_details": {
        "name": "BM.GPU.H200.8",
        "gpu_specs": {
          "gpu_memory_in_gbs": 1128,
          "gpu_count": 8,
          "gpu_type": "H200",
          "quantization": ["awq", "gptq", "fp8", "int8", "gguf"],
          "ranking": {"cost": 100, "performance": 110}
        }
      },
      "configurations": [
        {
          "model_details": {
            "model_size_gb": 140.0,
            "kv_cache_size_gb": 26.8,
            "total_model_gb": 166.8
          },
          "deployment_params": {
            "quantization": "bfloat16",
            "max_model_len": 131072,
            "batch_size": 1
          },
          "recommendation": "Model fits well within the allowed compute shape..."
        }
      ]
    }
  ],
  "troubleshoot": ""
}
```

### Ranking Fields

| Field | Description |
|---|---|
| `ranking.cost` | Higher = more cost-effective (lower price for memory provided) |
| `ranking.performance` | Higher = better throughput (H100/H200 > A100 > A10) |

Results are sorted by `performance` descending, so the first recommendation is the highest-performance shape that fits the model.

### Limitations

- Only supports **Safetensor format** (requires `config.json` + `.safetensors` files on HF)
- Only supports **decoder-only** text generation models
- Does not support GGUF models (use llama.cpp container + manual sizing)
- Does not support encoder-only (BERT) or encoder-decoder (T5, Gemma encoder) models

### Troubleshooting

| Error in `troubleshoot` field | Cause | Fix |
|---|---|---|
| `"Please provide a model in Safetensor format"` | GGUF or other format | Use a HF model with `config.json` + `.safetensors` |
| `"Please provide a decoder-only text-generation model"` | Encoder/encoder-decoder model | Use LLaMA, Falcon, Mistral, etc. |
