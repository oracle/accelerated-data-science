# Recommender Operator

Recommender Systems are designed to suggest relevant items, products, or content to users based on their preferences and behaviors. These systems are widely used in various industries such as e-commerce, entertainment, and social media to enhance user experience by providing personalized recommendations. They help in increasing user engagement, satisfaction, and sales by predicting what users might like or need based on their past interactions and the preferences of similar users.


Below are the steps to configure and run the Recommender Operator on different resources.

## 1. Prerequisites

Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation. This step is mandatory as it sets up default values for different options while running the Recommender Operator on OCI Data Science jobs or OCI Data Flow applications. If you have previously done this and used a flexible shape, make sure to adjust `ml_job_config.ini` with shape config details and `docker_registry` information.

- ocpus = 1
- memory_in_gbs = 16
- docker_registry = `<iad.ocir.io/namespace/>`

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads operator init -t recommender --overwrite --output ~/recommender/
```

The most important files expected to be generated are:

- `recommender.yaml`: Contains recommender related configuration.
- `backend_operator_local_python_config.yaml`: This includes a local backend configuration for running recommender in a local environment. The environment should be set up manually before running the operator.
- `backend_operator_local_container_config.yaml`: This includes a local backend configuration for running recommender within a local container. The container should be built before running the operator. Please refer to the instructions below for details on how to accomplish this.
- `backend_job_container_config.yaml`: Contains Data Science job-related config to run recommender in a Data Science job within a container (BYOC) runtime. The container should be built and published before running the operator. Please refer to the instructions below for details on how to accomplish this.
- `backend_job_python_config.yaml`: Contains Data Science job-related config to run recommender in a Data Science job within a conda runtime. The conda should be built and published before running the operator.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running recommender on the local conda environment

To run recommender locally, create and activate a new conda environment (`ads-recommender`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- report-creator
- cerberus
- "git+https://github.com/oracle/accelerated-data-science.git@feature/recommender#egg=oracle-ads"
```

Please review the previously generated `recommender.yaml` file using the `init` command, and make any necessary adjustments to the input and output file locations. By default, it assumes that the files should be located in the same folder from which the `init` command was executed.

Use the command below to verify the recommender config.

```bash
ads operator verify -f ~/recommender/recommender.yaml
```

Use the following command to run the recommender within the `ads-recommender` conda environment.

```bash
ads operator run -f ~/recommender/recommender.yaml -b local
```

The operator will run in your local environment without requiring any additional modifications.

## 4. Running recommender on the local container

To run the recommender operator within a local container, follow these steps:

Use the command below to build the recommender container.

```bash
ads operator build-image -t recommender
```

This will create a new `recommender:v1` image, with `/etc/operator` as the designated working directory within the container.


Check the `backend_operator_local_container_config.yaml` config file. By default, it should have a `volume` section with the `.oci` configs folder mounted.

```yaml
volume:
  - "/Users/<user>/.oci:/root/.oci"
```

Mounting the OCI configs folder is only required if an OCI Object Storage bucket will be used to store the input recommender data or output recommender result. The input/output folders can also be mounted to the container.

```yaml
volume:
  - /Users/<user>/.oci:/root/.oci
  - /Users/<user>/recommender/data:/etc/operator/data
  - /Users/<user>/recommender/result:/etc/operator/result
```

The full config can look like:
```yaml
kind: operator.local
spec:
  image: recommender:v1
  volume:
  - /Users/<user>/.oci:/root/.oci
  - /Users/<user>/recommender/data:/etc/operator/data
  - /Users/<user>/recommender/result:/etc/operator/result
type: container
version: v1
```

Run the recommender within a container using the command below:

```bash
ads operator run -f ~/recommender/recommender.yaml --backend-config ~/recommender/backend_operator_local_container_config.yaml
```

## 5. Running recommender in the Data Science job within container runtime

To execute the recommender operator within a Data Science job using container runtime, please follow the steps outlined below:

You can use the following command to build the recommender container. This step can be skipped if you have already done this for running the operator within a local container.

```bash
ads operator build-image -t recommender
```

This will create a new `recommender:v1` image, with `/etc/operator` as the designated working directory within the container.

Publish the `recommender:v1` container to the [Oracle Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/home.htm). To become familiar with OCI, read the documentation links posted below.

- [Access Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm#access)
- [Create repositories](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm#top)
- [Push images](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm#Pushing_Images_Using_the_Docker_CLI)

To publish `recommender:v1` to OCR, use the command posted below:

```bash
ads operator publish-image recommender:v1 --registry <iad.ocir.io/tenancy/>
```

After the container is published to OCR, it can be used within Data Science jobs service. Check the `backend_job_container_config.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain an image property, something like `image: iad.ocir.io/<tenancy>/recommender:v1`. More details about supported options can be found in the ADS Jobs documentation - [Run a Container](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_container.html).

Adjust the `recommender.yaml` config with proper input/output folders. When the recommender is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `recommender.yaml` and adjust the following fields:

```yaml
input_data:
  url: oci://bucket@namespace/recommender/input_data/data.csv
output_directory:
  url: oci://bucket@namespace/recommender/result/
```

Run the recommender on the Data Science jobs using the command posted below:

```bash
ads operator run -f ~/recommender/recommender.yaml --backend-config ~/recommender/backend_job_container_config.yaml
```

The logs can be monitored using the `ads opctl watch` command.

```bash
ads opctl watch <OCID>
```

## 6. Running recommender in the Data Science job within conda runtime

To execute the recommender operator within a Data Science job using conda runtime, please follow the steps outlined below:

You can use the following command to build the recommender conda environment.

```bash
ads operator build-conda -t recommender
```

This will create a new `recommender_v1` conda environment and place it in the folder specified within `ads opctl configure` command.

Use the command below to Publish the `recommender_v1` conda environment to the Object Storage bucket.

```bash
ads operator publish-conda -t recommender
```
More details about configuring CLI can be found here - [Configuring CLI](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html)


After the conda environment is published to Object Storage, it can be used within Data Science jobs service. Check the `backend_job_python_config.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain a `conda` section.

```yaml
conda:
  type: published
  uri: oci://bucket@namespace/conda_environments/cpu/recommender/1/recommender_v1
```

More details about supported options can be found in the ADS Jobs documentation - [Run a Python Workload](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_python.html).

Adjust the `recommender.yaml` config with proper input/output folders. When the recommender is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `recommender.yaml` and adjust the following fields:

```yaml
input_data:
  url: oci://bucket@namespace/recommender/input_data/data.csv
output_directory:
  url: oci://bucket@namespace/recommender/result/
test_data:
  url: oci://bucket@namespace/recommender/input_data/test.csv
```

Run the recommender on the Data Science jobs using the command posted below:

```bash
ads operator run -f ~/recommender/recommender.yaml --backend-config ~/recommender/backend_job_python_config.yaml
```

The logs can be monitored using the `ads opctl watch` command.

```bash
ads opctl watch <OCID>
```
