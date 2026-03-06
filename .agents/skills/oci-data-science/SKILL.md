---
name: oci-data-science
description: OCI Data Science service patterns including Jobs, Pipelines, Model Catalog, authentication, and the ADS SDK beyond AQUA. Triggered when user asks about OCI Data Science Jobs, ML Pipelines, model artifacts, conda environments, or general ADS SDK usage.
user-invocable: true
disable-model-invocation: false
---

# OCI Data Science with ADS SDK

Use this skill for general OCI Data Science service operations beyond AI Quick Actions (AQUA), including Jobs, Pipelines, Model Catalog, and authentication.

## Authentication

```python
import ads

# In OCI Notebook Sessions (recommended)
ads.set_auth("resource_principal")

# Local development with API key (~/.oci/config)
ads.set_auth("api_key")

# Local development with security token (session-based)
ads.set_auth("security_token")

# Get the current signer for raw OCI API calls
from ads.common.auth import default_signer
auth = default_signer()
```

## OCI Client Factory

```python
from ads.common.oci_client import OCIClientFactory

# Create service clients
ds_client = OCIClientFactory(**auth).data_science
os_client = OCIClientFactory(**auth).object_storage
compute_client = OCIClientFactory(**auth).compute
identity_client = OCIClientFactory(**auth).identity
```

## Data Science Jobs

Jobs allow running Python scripts, notebooks, or containers on managed infrastructure.

### Create and Run a Job

```python
from ads.jobs import Job, DataScienceJob, ScriptRuntime

job = (
    Job(name="my-training-job")
    .with_infrastructure(
        DataScienceJob()
        .with_compartment_id("ocid1.compartment.oc1..xxx")
        .with_project_id("ocid1.datascienceproject.oc1.iad.xxx")
        .with_shape_name("VM.Standard.E4.Flex")
        .with_shape_config_details(memory_in_gbs=16, ocpus=2)
        .with_block_storage_size(50)
        .with_log_group_id("ocid1.loggroup.oc1.iad.xxx")
        .with_log_id("ocid1.log.oc1.iad.xxx")
    )
    .with_runtime(
        ScriptRuntime()
        .with_source("path/to/script.py")
        .with_service_conda("pytorch21_p39_gpu_v1")
    )
)

job.create()
job_run = job.run()
job_run.watch()  # Stream logs
```

### Job with Container Runtime

```python
from ads.jobs import Job, DataScienceJob, ContainerRuntime

job = (
    Job(name="my-container-job")
    .with_infrastructure(
        DataScienceJob()
        .with_compartment_id("ocid1.compartment.oc1..xxx")
        .with_project_id("ocid1.datascienceproject.oc1.iad.xxx")
        .with_shape_name("VM.GPU.A10.1")
    )
    .with_runtime(
        ContainerRuntime()
        .with_image("iad.ocir.io/my-namespace/my-image:latest")
        .with_environment_variable(MY_VAR="value")
    )
)
```

### Job with Git Runtime

```python
from ads.jobs import Job, DataScienceJob, GitPythonRuntime

job = (
    Job(name="my-git-job")
    .with_infrastructure(
        DataScienceJob()
        .with_compartment_id("ocid1.compartment.oc1..xxx")
        .with_project_id("ocid1.datascienceproject.oc1.iad.xxx")
        .with_shape_name("VM.Standard.E4.Flex")
    )
    .with_runtime(
        GitPythonRuntime()
        .with_source("https://github.com/user/repo.git")
        .with_entry_point("train.py")
        .with_service_conda("generalml_p311_cpu_v1")
    )
)
```

### Watch Job Logs

```bash
ads opctl watch <job_run_ocid> --auth resource_principal
```

## ML Pipelines

Pipelines chain multiple steps into reproducible ML workflows.

### Create a Pipeline

```python
from ads.pipeline import Pipeline, PipelineStep, ScriptStep

step1 = (
    ScriptStep("data-prep")
    .with_infrastructure(
        DataScienceJob()
        .with_shape_name("VM.Standard.E4.Flex")
        .with_block_storage_size(50)
    )
    .with_runtime(
        ScriptRuntime()
        .with_source("prep_data.py")
        .with_service_conda("generalml_p311_cpu_v1")
    )
)

step2 = (
    ScriptStep("train-model")
    .with_infrastructure(
        DataScienceJob()
        .with_shape_name("VM.GPU.A10.1")
    )
    .with_runtime(
        ScriptRuntime()
        .with_source("train.py")
        .with_service_conda("pytorch21_p39_gpu_v1")
    )
)

pipeline = (
    Pipeline("my-ml-pipeline")
    .with_compartment_id("ocid1.compartment.oc1..xxx")
    .with_project_id("ocid1.datascienceproject.oc1.iad.xxx")
    .with_step_details([step1, step2])
    .with_dag(["data-prep >> train-model"])
    .with_log_group_id("ocid1.loggroup.oc1.iad.xxx")
)

pipeline.create()
pipeline_run = pipeline.run()
pipeline_run.watch()
```

## Model Catalog

Store, version, and manage ML models in a centralized catalog.

### Save a Model

```python
from ads.model import GenericModel

model = GenericModel(estimator=my_trained_model, artifact_dir="./model_artifact")
model.prepare(
    inference_conda_env="generalml_p311_cpu_v1",
    inference_python_version="3.11",
    model_file_name="model.pkl",
)
model.verify(test_data)
model.save(
    display_name="my-classification-model",
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
)
```

### Load a Model

```python
from ads.model import GenericModel

model = GenericModel.from_model_catalog(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",
    artifact_dir="./downloaded_artifact",
)
predictions = model.predict(test_data)
```

### Model Deployment (non-AQUA)

```python
from ads.model.deployment import ModelDeployment, ModelDeploymentContainerRuntime

deployment = (
    ModelDeployment()
    .with_display_name("my-model-deployment")
    .with_compartment_id("ocid1.compartment.oc1..xxx")
    .with_project_id("ocid1.datascienceproject.oc1.iad.xxx")
    .with_model_id("ocid1.datasciencemodel.oc1.iad.xxx")
    .with_shape_name("VM.Standard.E4.Flex")
    .with_shape_config_details(memory_in_gbs=16, ocpus=2)
    .with_bandwidth_mbps(10)
)
deployment.deploy()
deployment.predict(test_data)
```

## LangChain Integration

### Chat with OCI Model Deployment

```python
from ads.llm import ChatOCIModelDeployment

llm = ChatOCIModelDeployment(
    model="odsc-llm",
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<deployment_ocid>/predict",
    streaming=True,
)

response = llm.invoke("Tell me about Oracle Cloud.")
print(response.content)
```

### With Tool Calling (LangChain)

```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72F"

agent = create_tool_calling_agent(llm, [get_weather], prompt_template)
executor = AgentExecutor(agent=agent, tools=[get_weather])
result = executor.invoke({"input": "What's the weather in Austin?"})
```

## Object Storage Helpers

```python
from ads.common.object_storage_details import ObjectStorageDetails

# Parse an OCI Object Storage path
oss = ObjectStorageDetails.from_path("oci://my-bucket@my-namespace/path/to/file.csv")
print(oss.bucket)     # "my-bucket"
print(oss.namespace)  # "my-namespace"
print(oss.filepath)   # "path/to/file.csv"
```

## Conda Environments

OCI Data Science provides pre-built conda environments:

| Environment | Use Case |
|---|---|
| `generalml_p311_cpu_v1` | General ML (scikit-learn, XGBoost, etc.) |
| `pytorch21_p39_gpu_v1` | PyTorch GPU workloads |
| `tensorflow29_p39_gpu_v1` | TensorFlow GPU workloads |
| `pyspark32_p38_cpu_v3` | PySpark / big data |

List available:
```python
from ads.common.conda_environment import CondaEnvironment
envs = CondaEnvironment.list()
```

## Project Structure Reference

```
ads/                    # Main SDK source
├── aqua/              # AI Quick Actions (AQUA) - GenAI/LLM lifecycle
├── common/            # Shared: auth, OCI clients, serializers, utils
├── model/             # Model framework, artifacts, deployment
├── jobs/              # OCI Data Science Jobs
├── pipeline/          # ML Pipelines
├── opctl/             # CLI operators, log watching
├── llm/               # LangChain integration
├── feature_store/     # Feature Store
└── catalog/           # Model catalog helpers
```

## Key Documentation

- [ADS User Guide](https://accelerated-data-science.readthedocs.io/en/latest/index.html)
- [ADS API Reference](https://accelerated-data-science.readthedocs.io/en/latest/modules.html)
- [OCI Data Science Docs](https://docs.oracle.com/en-us/iaas/data-science/using/data-science.htm)
- [AQUA Docs](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm)
- [Samples Repo](https://github.com/oracle-samples/oci-data-science-ai-samples)
