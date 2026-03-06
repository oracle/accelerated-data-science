# Batch Inferencing with AQUA

Batch inferencing runs LLM inference over a dataset offline — no live model deployment needed. Uses an OCI Data Science **Job** with the vLLM container, writes outputs to Object Storage.

Use this pattern when:
- Running nightly inference over thousands of prompts
- Cost matters more than latency (no idle GPU time between requests)
- Results need to be stored, not streamed

## Architecture

```
[input.json in job artifact]
         ↓
  DataScienceJob (GPU shape)
  ContainerRuntime (dsmc://odsc-vllm-serving)
         ↓
  vllm_batch_inferencing.py
  - Loads model from HF Hub
  - Runs LLM.generate() on all prompts
  - Saves output_prompts.json
         ↓
  upload_folder() → OCI Object Storage
```

## Setup

### 1. Install Dependencies

```python
from ads.jobs import Job, DataScienceJob, ContainerRuntime
import ads
import os

ads.set_auth("resource_principal")
```

### 2. Configure Variables

```python
compartment_id = os.environ["PROJECT_COMPARTMENT_OCID"]
project_id = os.environ["PROJECT_OCID"]

log_group_id = "ocid1.loggroup.oc1.xxx.xxxxx"
log_id = "ocid1.log.oc1.xxx.xxxxx"

instance_shape = "VM.GPU.A10.2"
container_image = "dsmc://odsc-vllm-serving:0.6.2"

bucket = "<bucket_name>"           # must be a versioned bucket
namespace = "<bucket_namespace>"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_token = "<your-huggingface-token>"
```

### 3. Create Job Artifacts Directory

```python
import os

job_artifacts = "job-artifacts"
os.makedirs(job_artifacts, exist_ok=True)
```

### 4. Create `job-artifacts/input.json`

```json
{
    "vllm_engine_config": {
        "tensor_parallel_size": 2,
        "disable_custom_all_reduce": true
    },
    "sampling_config": {
        "max_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.85
    },
    "data": [
        [
            {
                "role": "system",
                "content": "You are a friendly chatbot who is a great story teller."
            },
            {
                "role": "user",
                "content": "Tell me a 1000 words story"
            }
        ]
    ]
}
```

**Fields:**
- `vllm_engine_config` — any [vLLM LLM class](https://docs.vllm.ai/en/latest/dev/offline_inference/llm.html) kwargs
- `sampling_config` — any [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html) kwargs
- `data` — list of message arrays (OpenAI chat format)

### 5. Create `job-artifacts/vllm_batch_inferencing.py`

```python
from typing import Any, Dict, List
import os, json, logging
from ads.model.datascience_model import DataScienceModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from ads.common.utils import ObjectStorageDetails
from ads.aqua.common.utils import upload_folder


class Constants:
    OCI_IAM_TYPE = "OCI_IAM_TYPE"
    INPUT_PROMPT = "input_prompt"
    OUTPUT = "generated_output"
    VLLM_ENGINE_CONFIG = "vllm_engine_config"
    SAMPLING_CONFIG = "sampling_config"
    MODEL = "MODEL"
    HF_TOKEN = "HF_TOKEN"
    OUTPUT_FOLDER_PATH = os.path.expanduser("~/outputs")
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "output_prompts.json")
    OS_OBJECT = "batch-inference"


class Deployment:
    def __init__(self, vllm_config: Dict[str, Any]) -> None:
        hf_token = os.environ.get(Constants.HF_TOKEN, "")
        if hf_token:
            login(token=hf_token)
        self.model = os.environ[Constants.MODEL]
        self.vllm_config = vllm_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.llm = LLM(model=self.model, **self.vllm_config)

    def requests(self, data: List, sampling_config: Dict = None) -> List[Dict]:
        prompt_token_ids = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            for messages in data
        ]
        sampling_params = (
            SamplingParams(**sampling_config)
            if sampling_config
            else SamplingParams(max_tokens=250, temperature=0.6, top_p=0.9)
        )
        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )
        processed = []
        for output in outputs:
            processed.append({
                Constants.INPUT_PROMPT: self.tokenizer.decode(output.prompt_token_ids),
                Constants.OUTPUT: output.outputs[0].text,
                Constants.VLLM_ENGINE_CONFIG: self.vllm_config,
                Constants.SAMPLING_CONFIG: sampling_config,
            })
        os.makedirs(Constants.OUTPUT_FOLDER_PATH, exist_ok=True)
        with open(Constants.OUTPUT_FILE_PATH, "w") as f:
            json.dump(processed, f)
        return processed


def main():
    with open("job-artifacts/input.json") as f:
        input_data = json.load(f)

    deployment = Deployment(input_data[Constants.VLLM_ENGINE_CONFIG])
    deployment.requests(input_data["data"], input_data.get(Constants.SAMPLING_CONFIG))

    prefix = os.environ.get("PREFIX", Constants.OS_OBJECT)
    os_path = ObjectStorageDetails(os.environ["BUCKET"], os.environ["NAMESPACE"], prefix).path
    print(upload_folder(os_path=os_path, local_dir=Constants.OUTPUT_FOLDER_PATH, model_name="outputs"))


if __name__ == "__main__":
    main()
```

## Define and Run the Job

### Infrastructure

```python
infrastructure = (
    DataScienceJob()
    .with_log_group_id(log_group_id)
    .with_log_id(log_id)
    .with_job_infrastructure_type("ME_STANDALONE")
    .with_compartment_id(compartment_id)
    .with_project_id(project_id)
    .with_shape_name(instance_shape)
    .with_block_storage_size(80)  # GB; minimum 50
)
```

### Container Runtime

```python
container_runtime = (
    ContainerRuntime()
    .with_image(container_image)
    .with_environment_variable(
        HF_TOKEN=hf_token,
        MODEL=model_name,
        BUCKET=bucket,
        NAMESPACE=namespace,
        OCI_IAM_TYPE="resource_principal",
        PREFIX="batch-inference",
    )
    .with_entrypoint(["bash", "-c"])
    .with_cmd(
        "microdnf install -y unzip && "
        "pip install oracle-ads[opctl] && "
        "cd /home/datascience/ && "
        "unzip job-artifacts.zip -d . && "
        "chmod +x job-artifacts/vllm_batch_inferencing.py && "
        "python job-artifacts/vllm_batch_inferencing.py"
    )
    .with_artifact("job-artifacts")
)
```

### Create and Run

```python
job = (
    Job(name=f"Batch inferencing - {model_name}")
    .with_infrastructure(infrastructure)
    .with_runtime(container_runtime)
)
job.create()
run = job.run()
run.watch()   # stream logs
```

## Download Output

```bash
oci os object bulk-download \
  --bucket-name <bucket> \
  --namespace <namespace> \
  --prefix batch-inference \
  --dest-dir . \
  --auth resource_principal
```

The output file is at `batch-inference/outputs/output_prompts.json`. Each entry contains:
- `input_prompt` — decoded tokenized input
- `generated_output` — model response text
- `vllm_engine_config` — engine config used
- `sampling_config` — sampling params used

## Shape Selection for Batch Jobs

| Model | Shape | `tensor_parallel_size` |
|---|---|---|
| 7B–8B | VM.GPU.A10.2 | 2 |
| 13B | BM.GPU.A10.4 | 4 |
| 70B | BM.GPU.A100-v2.8 | 8 |

Set `tensor_parallel_size` in `vllm_engine_config` to match the GPU count of your shape.
