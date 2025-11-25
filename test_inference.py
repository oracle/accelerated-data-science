import json
import requests
import ads

# Set up OCI security token authentication
ads.set_auth("security_token")

# Your Model Deployment OCID and endpoint URL
md_ocid = "ocid1.datasciencemodeldeploymentint.oc1.iad.amaaaaaav66vvniasakhgqe4hk6eqgci7jmj2nvxjzldaqlnb7ji7vjr5p6a"
endpoint = "https://modeldeployment-int.us-ashburn-1.oci.oc-test.com/ocid1.datasciencemodeldeploymentint.oc1.iad.amaaaaaav66vvniasakhgqe4hk6eqgci7jmj2nvxjzldaqlnb7ji7vjr5p6a/predict"

# OCI request signer
auth = ads.common.auth.default_signer()["signer"]


def predict(model_name):
    predict_data = {
        "model": model_name,
        "prompt": "[user] Write a SQL query to answer the question based on the table schema.\n\ncontext: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\nquestion: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "max_tokens": 100,
        "temperature": 0,
    }
    predict_headers = {"cx": "application/json", "opc-request-id": "test-id"}
    response = requests.post(
        endpoint,
        headers=predict_headers,
        data=json.dumps(predict_data),
        auth=auth,
        verify=False,  # Use verify=True in production!
    )
    print("Status:", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Response.text:", response.text)


if __name__ == "__main__":
    ft_model_name = "my-llama-v3.1-8b-instruct-ft"
    print(f"Testing FT model: {ft_model_name}")
    predict(ft_model_name)
