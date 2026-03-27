---
name: aqua-troubleshooting
description: Diagnose and fix OCI AI Quick Actions (AQUA) issues including deployment failures, OOM errors, authorization problems, capacity issues, container errors, and policy misconfigurations. Triggered when user encounters errors or needs help debugging AQUA workflows.
user-invocable: true
disable-model-invocation: false
---

# AQUA Troubleshooting Guide

Use this skill when the user encounters errors or needs help diagnosing issues with OCI AI Quick Actions deployments, fine-tuning, evaluation, or model registration.

## Step 1: Check Logs

Always check logs first. Logging must be enabled during deployment creation.

```bash
# Watch live logs for a model deployment
ads opctl watch <model_deployment_ocid> --auth resource_principal

# Watch logs for a job run (fine-tuning/evaluation)
ads opctl watch <job_run_ocid> --auth resource_principal
```

To get the OCID: AQUA > Model Deployments tab > click deployment > copy OCID from details.

## Common Deployment Errors

### 1. Service Timeout Error

**Symptom**: Model deployment fails during startup - couldn't load the model in time.

**Diagnosis**: Check logs via `ads opctl watch`.

**Solutions**:
- The model may be too large for the selected shape
- Try a larger GPU shape
- Reduce `--max-model-len` to decrease memory requirements

### 2. Out of Memory (OOM) Error

#### Case A: Model Too Large for GPU

**Symptom**: CUDA OOM error during model loading.

**Solutions** (try in order):
1. **Use a bigger shape** (more GPU memory)
2. **Try FP8 quantization**: Add `--quantization fp8` to `PARAMS`
3. **Try 4-bit quantization**: Add `--quantization bitsandbytes --load-format bitsandbytes` to `PARAMS`
4. **Reduce context length**: Add `--max-model-len <smaller_value>` to `PARAMS`

```python
# Example: Deploy with quantization to fit on smaller GPU
env_var={
    "PARAMS": "--quantization fp8 --max-model-len 4096",
}
```

#### Case B: KV Cache Too Small

**Symptom**: Error says "max seq len is larger than maximum tokens in KV cache".

**Solution**: The error log contains a hint for the max supported `--max-model-len`. Set it to that value:
```python
env_var={
    "PARAMS": "--max-model-len <value_from_log>",
}
```

### 3. Trust Remote Code Error

**Symptom**: Error mentions `trust_remote_code=True` is required.

**Solution**: Add `--trust-remote-code` to PARAMS (leave value blank):
```python
env_var={
    "PARAMS": "--trust-remote-code --max-model-len 4096",
}
```

### 4. Architecture Not Supported

**Symptom**: `ValueError: Model architectures ['<NAME>'] are not supported for the current vLLM instance.`

**Solutions**:
1. Check [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
2. If not supported by vLLM, use the BYOC (Bring Your Own Container) approach
3. For some models, add `--trust-remote-code`

### 5. Capacity Issues

**Symptom**: "No capacity for the specified shape" or "Out of host capacity".

**Solutions**:
1. Try a different availability domain
2. Try a different GPU shape
3. Use [capacity reservations](https://docs.oracle.com/en-us/iaas/data-science/using/gpu-using.htm#gpu-use-reserve)
4. Wait and retry (capacity is dynamic)

## Authorization Errors

### Root Causes

Authorization errors arise from:
1. **Missing OCI IAM policies**
2. **Object Storage bucket without versioning enabled**
3. **Notebook session not in the same compartment** as the dynamic group

### Required Policies

Set up policies via Oracle Resource Manager (ORM) - recommended:
```
# Go to: AQUA > Policies > Setup via ORM
```

Or verify with the AQUA Policy Verification tool:
```python
from ads.aqua.verify_policies import AquaVerifyPoliciesApp
verify_app = AquaVerifyPoliciesApp()
result = verify_app.verify()
```

### Policy-to-Operation Mapping

| Operation | Required Policy |
|---|---|
| Create/List Models | `manage data-science-models` in compartment |
| Create/List Deployments | `manage data-science-model-deployments` in compartment |
| Create/List Model Version Sets | `manage data-science-modelversionsets` in compartment |
| Create/List Jobs (FT/Eval) | `manage data-science-job-runs` in compartment |
| Read Object Storage | `read buckets` + `read objectstorage-namespaces` in compartment |
| Write Object Storage | `manage object-family` in compartment |
| List Log Groups | `use logging-family` in compartment |
| Use Private Endpoints | `use virtual-network-family` in compartment |
| Tag Resources | `use tag-namespaces in tenancy` |
| Evaluation/Fine-Tuning | `manage data-science-models` + `read resource-availability` + `use virtual-network-family` |

### Bucket Versioning

Object Storage bucket **must** have versioning enabled:
```bash
# Check versioning status
oci os bucket get -bn <bucket-name> --auth resource_principal | jq ".data.versioning"
# Should return "Enabled"
```

## Environment Setup Issues

### Authentication

```python
import ads

# In OCI Notebook Sessions
ads.set_auth("resource_principal")

# Local development with API key
ads.set_auth("api_key")

# Local development with security token
ads.set_auth("security_token")
```

### Required Environment Variables (for local/internal development)

```bash
export OCI_IAM_TYPE="security_token"
export OCI_CONFIG_PROFILE=<your-profile>
export OCI_ODSC_SERVICE_ENDPOINT="https://datascience.us-ashburn-1.oci.oraclecloud.com"
```

### HuggingFace Gated Models

```bash
export HF_TOKEN=<your_hf_read_token>
# OR
huggingface-cli login
```

## Fine-Tuning Specific Issues

### Dataset Format Errors

- Ensure JSONL format (one valid JSON per line)
- All rows must have same schema
- For instruction format: `prompt` and `completion` keys required
- For conversational format: `messages` key with `role`/`content` objects
- Verify no trailing commas or invalid JSON

### Distributed Training Failures

- VCN + Subnet required for `replica > 1`
- Logging required for distributed training
- Multi-node overhead is significant; single replica with multi-GPU shape is preferred
- Check that all nodes can communicate (security lists / NSGs allow traffic)

## Evaluation Specific Issues

### Evaluation Job Fails

- Ensure deployment is in `ACTIVE` state before running evaluation
- Dataset must be JSONL with `prompt` and `completion` keys
- Report path must be writable Object Storage location
- Block storage size must be sufficient (default: 50 GB)

### BERTScore Issues

- BERTScore is not suitable for evaluating code generation tasks
- Consider ROUGE for summarization-focused evaluations
- The evaluation model endpoint must be reachable from the evaluation job

## Diagnostic Commands

```bash
# Check deployment status
ads aqua deployment get --model_deployment_id <ocid>

# List all deployments (check for failed ones)
ads aqua deployment list --compartment_id <compartment_ocid>

# Check model details
ads aqua model get --model_id <model_ocid>

# Verify policies
ads aqua verify_policies
```

## Key Source Files

- `ads/aqua/verify_policies/` — Policy verification app
- `ads/aqua/common/errors.py` — Error hierarchy (AquaValueError, AquaRuntimeError, etc.)
- `ads/aqua/training/exceptions.py` — Training job exit code mappings
- `ads/aqua/extension/errors.py` — HTTP error message templates
