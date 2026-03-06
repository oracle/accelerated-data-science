# LMCache — KV Cache Persistence for AQUA Deployments

LMCache persists the KV cache across requests to CPU RAM, dramatically reducing repeated computation for multi-turn workloads. Useful when many sessions share long system prompts or conversation prefixes.

## When to Use LMCache

- Multi-turn chat with long conversation histories
- Shared system prompts served to many users
- RAG scenarios with large repeated context prefixes
- Any workload where prompts share common prefixes

## vLLM Configuration

Add to the `PARAMS` environment variable (under **Show Advanced Options** → **Model deployment configuration**):

```
--max-model-len 24000 --gpu-memory-utilization 0.90 --kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}
```

> **Important:** No spaces inside the `--kv-transfer-config` JSON value.

## Required Environment Variables

| Name | Value | Purpose |
|---|---|---|
| `PYTHONHASHSEED` | `0` | Deterministic hashing for cache key generation |
| `LMCACHE_TRACK_USAGE` | `false` | Disables usage tracking |
| `LMCACHE_CHUNK_SIZE` | `256` | KV cache chunk size in tokens |
| `LMCACHE_LOCAL_CPU` | `True` | Use CPU RAM as cache backend |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `50` | Maximum CPU cache size in GB |

## Python SDK Example

```python
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

details = CreateModelDeploymentDetails(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",   # ibm-granite/granite-3.3-8b-instruct
    instance_shape="VM.GPU.A10.1",
    display_name="granite-lmcache",
    log_group_id="ocid1.loggroup.oc1.iad.xxx",
    log_id="ocid1.log.oc1.iad.xxx",
    env_var={
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions",
        "PARAMS": '--max-model-len 24000 --gpu-memory-utilization 0.90 --kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        "PYTHONHASHSEED": "0",
        "LMCACHE_TRACK_USAGE": "false",
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "50",
    },
)
deployment = AquaDeploymentApp().create(create_deployment_details=details)
```

## CLI Example

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.1" \
  --display_name "granite-lmcache" \
  --env_var '{
    "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions",
    "PARAMS": "--max-model-len 24000 --gpu-memory-utilization 0.90 --kv-transfer-config {\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}",
    "PYTHONHASHSEED": "0",
    "LMCACHE_TRACK_USAGE": "false",
    "LMCACHE_CHUNK_SIZE": "256",
    "LMCACHE_LOCAL_CPU": "True",
    "LMCACHE_MAX_LOCAL_CPU_SIZE": "50"
  }'
```

## Sizing Guidelines

Default LMCache CPU size is 5 GB. Size `LMCACHE_MAX_LOCAL_CPU_SIZE` to hold the typical KV "working set" plus overhead:

| Model Size | Recommended CPU Cache |
|---|---|
| 7B–13B | 20–50 GB |
| 70B | 50–100 GB |

Refine sizing with Prometheus metrics after initial deployment (see `.agents/skills/aqua-metrics/SKILL.md`).

## Requirements

- vLLM container only (`odsc-vllm-serving`) — LMCache is not supported with TGI or llama.cpp
- Requires VM.GPU.A10.1 or larger
- CPU RAM on the host must be sufficient for `LMCACHE_MAX_LOCAL_CPU_SIZE`
