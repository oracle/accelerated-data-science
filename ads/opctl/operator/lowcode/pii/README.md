# PII Operator


The PII Operator aims to detect and redact Personally Identifiable Information (PII) in datasets. PII data includes information such as names, addresses, and social security numbers, which can be used to identify individuals. This operator combine pattern matching and machine learning solution to identify PII, and then redacts or anonymizes it to protect the privacy of individuals.

Below are the steps to configure and run the PII Operator on different resources.

## 1. Prerequisites

Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation. This step is mandatory as it sets up default values for different options while running the PII Operator on OCI Data Science jobs.

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads operator init -t pii --overwrite --output ~/pii/
```

The most important files expected to be generated are:

- `pii.yaml`: Contains pii-related configuration.
- `pii_operator_local_python.yaml`: This includes a local backend configuration for running pii operator in a local environment. The environment should be set up manually before running the operator.
- `pii_operator_local_container.yaml`: This includes a local backend configuration for running pii operator within a local container. The container should be built before running the operator. Please refer to the instructions below for details on how to accomplish this.
- `pii_job_container.yaml`: Contains Data Science job-related config to run pii operator in a Data Science job within a container (BYOC) runtime. The container should be built and published before running the operator. Please refer to the instructions below for details on how to accomplish this.
- `pii_job_python.yaml`: Contains Data Science job-related config to run pii operator in a Data Science job within a conda runtime. The conda should be built and published before running the operator.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running Pii on the local conda environment

To run pii operator locally, create and activate a new conda environment (`ads-pii`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- aiohttp
- report-creator
- gender_guesser
- nameparser
- oracle_ads[opctl]
- plotly
- scrubadub
- scrubadub_spacy
- spacy-transformers==1.2.5
- spacy==3.6.1
```

Please review the previously generated `pii.yaml` file using the `init` command, and make any necessary adjustments to the input and output file locations. By default, it assumes that the files should be located in the same folder from which the `init` command was executed.

Use the command below to verify the pii config.

```bash
ads operator verify -f ~/pii/pii.yaml
```

Use the following command to run the pii operator within the `ads-pii` conda environment.

```bash
ads operator run -f ~/pii/pii.yaml -b local
```

The operator will run in your local environment without requiring any additional modifications.

## 4. Running pii on the local container

To run the pii operator within a local container, follow these steps:

Use the command below to build the pii container.

```bash
ads operator build-image -t pii
```

This will create a new `pii:v1` image, with `/etc/operator` as the designated working directory within the container.


Check the `pii_operator_local_container.yaml` config file. By default, it should have a `volume` section with the `.oci` configs folder mounted.

```yaml
volume:
  - "/Users/<user>/.oci:/root/.oci"
```

Mounting the OCI configs folder is only required if an OCI Object Storage bucket will be used to store the input data or output result. The input/output folders can also be mounted to the container.

```yaml
volume:
  - /Users/<user>/.oci:/root/.oci
  - /Users/<user>/pii/data:/etc/operator/data
  - /Users/<user>/pii/result:/etc/operator/result
```

The full config can look like:
```yaml
kind: operator.local
spec:
  image: pii:v1
  volume:
  - /Users/<user>/.oci:/root/.oci
  - /Users/<user>/pii/data:/etc/operator/data
  - /Users/<user>/pii/result:/etc/operator/result
type: container
version: v1
```

Run the pii operator within a container using the command below:

```bash
ads operator run -f ~/pii/pii.yaml --backend-config ~/pii/pii_operator_local_container.yaml
```

## 5. Running pii in the Data Science job within container runtime

To execute the pii operator within a Data Science job using container runtime, please follow the steps outlined below:

You can use the following command to build the forecast container. This step can be skipped if you have already done this for running the operator within a local container.

```bash
ads operator build-image -t pii
```

This will create a new `pii:v1` image, with `/etc/operator` as the designated working directory within the container.

Publish the `pii:v1` container to the [Oracle Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/home.htm). To become familiar with OCI, read the documentation links posted below.

- [Access Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm#access)
- [Create repositories](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm#top)
- [Push images](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm#Pushing_Images_Using_the_Docker_CLI)

To publish `pii:v1` to OCR, use the command posted below:

```bash
ads operator publish-image pii:v1 --registry <iad.ocir.io/tenancy/>
```

After the container is published to OCR, it can be used within Data Science jobs service. Check the `backend_job_container_config.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain an image property, something like `image: iad.ocir.io/<tenancy>/pii:v1`. More details about supported options can be found in the ADS Jobs documentation - [Run a Container](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_container.html).

Adjust the `pii.yaml` config with proper input/output folders. When the operator is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `pii.yaml` and adjust the following fields:

```yaml
input_data:
  url: oci://bucket@namespace/pii/input_data/data.csv
output_directory:
  url: oci://bucket@namespace/pii/result/
```

Run the pii operator on the Data Science jobs using the command posted below:

```bash
ads operator run -f ~/pii/pii.yaml --backend-config ~/pii/pii_job_container.yaml
```

The logs can be monitored using the `ads opctl watch` command.

```bash
ads opctl watch <OCID>
```


## 6. Running pii in the Data Science job within conda runtime

To execute the pii operator within a Data Science job using conda runtime, please follow the steps outlined below:

You can use the following command to build the pii conda environment.

```bash
ads operator build-conda -t pii
```

This will create a new `pii_v1` conda environment and place it in the folder specified within `ads opctl configure` command.

Use the command below to Publish the `pii_v1` conda environment to the Object Storage bucket.

```bash
ads opctl conda publish pii_v1
```
More details about configuring CLI can be found here - [Configuring CLI](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html)


After the conda environment is published to Object Storage, it can be used within Data Science jobs service. Check the `pii_job_python.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain a `conda` section.

```yaml
conda:
  type: published
  uri: oci://bucket@namespace/conda_environments/cpu/pii/1/pii_v1
```

More details about supported options can be found in the ADS Jobs documentation - [Run a Python Workload](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_python.html).

Adjust the `pii.yaml` config with proper input/output folders. When the pii is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `pii.yaml` and adjust the following fields:

```yaml
input_data:
  url: oci://bucket@namespace/pii/input_data/data.csv
output_directory:
  url: oci://bucket@namespace/pii/result/
```

Run the pii on the Data Science jobs using the command posted below:

```bash
ads operator run -f ~/pii/pii.yaml --backend-config ~/pii/pii_job_python.yaml
```

The logs can be monitored using the `ads opctl watch` command.

```bash
ads opctl watch <OCID>
```
