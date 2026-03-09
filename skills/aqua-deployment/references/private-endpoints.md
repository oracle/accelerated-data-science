# Model Deployment Private Endpoints

Private endpoints restrict inference traffic to a customer's private network — all public internet access is rejected. Required for enterprise customers with compliance mandates against public routing.

## Prerequisite Order

1. **Create the private endpoint** (Data Science → Private Endpoints in OCI Console)
2. **Create the model deployment** referencing that private endpoint

These steps must be done in this order. A private endpoint cannot be attached to an existing deployment.

## Creating the Private Endpoint

In the OCI Console:

1. Navigate to **Data Science** → **Private endpoints**
2. Click **Create private endpoint**
3. (Optional) Enter a name and description
4. Select the **VCN** for private access
5. Select the **subnet** containing the private endpoint
6. (Optional) Enter a subdomain (max 60 characters)
7. For **Resource Type**, select `MODEL_DEPLOYMENT`
8. Click **Create**

## Attaching to a Deployment (Console)

When creating a deployment via AI Quick Actions:

1. Go to **AI Quick Actions** → **Deployments** → **Create deployment**
2. Select model, shape, logging
3. Click **Show advanced options**
4. Under **Inference mode**, select `v1/chat/completions`
5. Under **Private endpoint**, select the endpoint from the dropdown
6. Click **Deploy**

## Attaching to a Deployment (Python SDK)

```python
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

details = CreateModelDeploymentDetails(
    model_id="ocid1.datasciencemodel.oc1.iad.xxx",
    instance_shape="VM.GPU.A10.2",
    display_name="private-deployment",
    private_endpoint_id="ocid1.datascienceprivateendpoint.oc1.iad.xxx",
    env_var={
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions",
    },
)
deployment = AquaDeploymentApp().create(create_deployment_details=details)
```

## Attaching to a Deployment (CLI)

```bash
ads aqua deployment create \
  --model_id "ocid1.datasciencemodel.oc1.iad.xxx" \
  --instance_shape "VM.GPU.A10.2" \
  --display_name "private-deployment" \
  --private_endpoint_id "ocid1.datascienceprivateendpoint.oc1.iad.xxx" \
  --env_var '{"MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/chat/completions"}'
```

## Invoking a Private Deployment

Use the **FQDN** from the private endpoint information page — not the standard model deployment URL.

```bash
# Via oci-cli
oci raw-request \
  --http-method POST \
  --target-uri "<FQDN>/<model-deployment-ocid>/predict" \
  --request-body '{
    "model": "odsc-llm",
    "prompt": "What are activation functions?",
    "max_tokens": 250,
    "temperature": 0.7,
    "top_p": 0.8
  }' \
  --auth resource_principal
```

```python
# Via Python requests
import oci
import requests

endpoint_fqdn = "https://<fqdn-from-private-endpoint-page>"
model_deployment_id = "ocid1.datasciencemodeldeployment.oc1.iad.xxx"

response = requests.post(
    f"{endpoint_fqdn}/{model_deployment_id}/predict",
    json={
        "model": "odsc-llm",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 500,
    },
    auth=oci.auth.signers.get_resource_principals_signer(),
)
print(response.json())
```

## Key Difference: FQDN vs Standard URL

| | Standard Deployment | Private Endpoint Deployment |
|---|---|---|
| URL format | `https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict` | `https://<FQDN>/<ocid>/predict` |
| Network path | Public internet | Private VCN only |
| IAM auth | Same (OCI signatures) | Same (OCI signatures) |

The FQDN is visible in the OCI Console on the private endpoint details page.

## Reference

OCI docs: https://docs.oracle.com/en-us/iaas/data-science/using/pe-network.htm
