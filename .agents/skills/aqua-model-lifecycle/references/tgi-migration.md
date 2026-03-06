# Migrating Models from TGI to vLLM

TGI (Text Generation Inference) is deprecated in AQUA. Model deployments using TGI-registered models will fail on restart. This guide shows how to update existing registrations.

## Affected Models

The following service-managed models were registered with the TGI container and need updating:

1. codegemma-1.1-2b
2. codegemma-1.1-7b-it
3. codegemma-2b
4. codegemma-7b
5. falcon-40b-instruct
6. gemma-1.1-7b-it
7. gemma-2b
8. gemma-2b-it
9. gemma-7b

**Impact:** Any deployment backed by these models will fail on restart until updated.

## Mitigation 1 — Update the Existing Registered Model

This preserves the existing model OCID and updates its container mapping.

### Step 1: Get the Current Model

```python
import oci
from ads import set_auth
from oci.auth.signers import get_resource_principals_signer
from oci.data_science import DataScienceClient

set_auth(auth='resource_principal')
resource_principal = get_resource_principals_signer()
client = DataScienceClient(config={}, signer=resource_principal)

model = client.get_model(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx"
)
print(model.data.custom_metadata_list)
```

### Step 2: Update `deployment-container` to vLLM

**Critical:** Copy ALL existing `custom_metadata_list` entries — only change the `deployment-container` value. Dropping any metadata will break other AQUA functionality.

```python
update_response = client.update_model(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",
    update_model_details=oci.data_science.models.UpdateModelDetails(
        custom_metadata_list=[
            oci.data_science.models.Metadata(
                category="Other",
                description="Deployment container mapping for SMC",
                key="deployment-container",
                value="odsc-vllm-serving",          # changed from odsc-tgi-serving
                has_artifact=False,
            ),
            oci.data_science.models.Metadata(
                category="Other",
                description="Fine-tuning container mapping for SMC",
                key="finetune-container",
                value="odsc-llm-fine-tuning",        # preserve as-is
                has_artifact=False,
            ),
            oci.data_science.models.Metadata(
                category="Other",
                description="model by reference flag",
                key="modelDescription",
                value="true",                        # preserve as-is
                has_artifact=False,
            ),
            oci.data_science.models.Metadata(
                category="Other",
                description="artifact location",
                key="artifact_location",
                value="oci://bucket@namespace/path/to/model",  # preserve as-is
                has_artifact=False,
            ),
            oci.data_science.models.Metadata(
                category="Other",
                description="Evaluation container mapping for SMC",
                key="evaluation-container",
                value="odsc-llm-evaluate",           # preserve as-is
                has_artifact=False,
            ),
        ]
    ),
)
print(update_response.data)
```

### Step 3: Create a New Deployment

After the update, wait a few seconds for propagation, then create a new deployment. It will automatically use the vLLM container.

```python
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

deployment = AquaDeploymentApp().create(
    CreateModelDeploymentDetails(
        model_id="ocid1.datasciencemodel.oc1.iad.xxx",
        instance_shape="VM.GPU.A10.2",
        display_name="gemma-2b-vllm",
    )
)
```

## Mitigation 2 — Re-Register the Model

Register a fresh copy of the model. After TGI deprecation, AQUA will automatically select the vLLM container for newly registered service models.

```bash
ads aqua model register \
  --model "google/gemma-2b-it" \
  --os_path "oci://my-bucket@my-namespace/models/gemma-2b-it/" \
  --inference_container "odsc-vllm-serving" \
  --download_from_hf True
```

Use this approach when:
- You want a clean model entry
- The original OCID does not need to be preserved
- Downstream deployments/pipelines can be updated to the new OCID

## Verification

To confirm the update was applied:

```python
model = client.get_model(model_id="ocid1.datasciencemodel.oc1.iad.xxx")
for meta in model.data.custom_metadata_list:
    if meta.key == "deployment-container":
        print(meta.value)  # should be "odsc-vllm-serving"
```

## References

- `ads/aqua/model/model.py` — `AquaModelApp.register()` for fresh registration
- OCI SDK docs: https://docs.oracle.com/en-us/iaas/tools/python-sdk-examples/2.162.0/datascience/update_model.py.html
