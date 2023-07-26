# Forecasting Operator

The Forecasting Operator leverages historical time series data to generate accurate forecasts for future trends. This operator aims to simplify and expedite the data science process by automating the selection of appropriate models and hyperparameters, as well as identifying relevant features for a given prediction task.

Below are the steps to configure and run the Forecasting Operator on different resources.

## 1. Prerequisites

Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation. This step is mandatory as it sets up default values for different options while running the Forecasting Operator on OCI Data Science jobs or OCI Data Flow applications. If you have previously done this and used a flexible shape, make sure to adjust `ml_job_config.ini` with shape config details and `docker_registry` information.

- ocpus = 1
- memory_in_gbs = 16
- docker_registry = `<iad.ocir.io/namespace/>`

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads opctl operator init -n forecast --overwrite --output /dst/configs/
```

The most important files expected to be generated are:

- `forecast.yaml`: Contains forecast-related configuration.
- `backend_operator_local_container_config.yaml`: Contains a local backend configuration to run forecasting in a local container.
- `backend_job_container_config.yaml`: Contains Data Science job-related config to run forecasting in a Data Science job.

## 3. Running forecasting on the local conda environment

To run forecasting locally, create and activate a new conda environment (`ads-forecasting`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- prophet
- neuralprophet
- pmdarima
- statsmodels
- datapane
- cerberus
- json2table
- sktime
- optuna==2.9.0
- automlx-23.2.1-py38-none-any.whl
- oracle_ads-2.8.7b0-py3-none-any.whl
```

Check the previously generated `forecast.yaml` and adjust the input and output file locations.

Use the command below to verify the forecasting config.

```bash
ads opctl operator verify -f /dst/configs/forecast.yaml
```

Use the following command to run the forecasting within the `ads-forecasting` conda environment.

```bash
python -m ads.opctl.operator.lowcode.forecast -f /dst/configs/forecast.yaml
```

## 4. Running forecasting on the local container

To run the forecasting operator within a local container, follow these steps:

Use the command below to build the forecast container.

```bash
ads opctl operator build-image -n forecast
```

This will create a new `forecast:v1` image, and `/etc/operator` will be used as the working directory in the container.


Check the `backend_operator_local_container_config.yaml` config. By default, it should have a `volume` section with the `.oci` configs folder mounted.

```yaml
volume:
  - "/Users/dmcherka/.oci:/root/.oci"
```

Mounting the OCI configs folder is only required if an OCI Object Storage bucket will be used to store the input forecasting data or output forecasting result. The input/output folders can also be mounted to the container.

```yaml
volume:
  - "/etc/forecast/data:/etc/operator/data"
  - "/etc/forecast/result:/etc/operator/result"
  - "/etc/.oci:/root/.oci"
```

Run the forecasting within a container using the command below:

```bash
ads opctl apply -f /dst/configs/forecast.yaml --backend-config /dst/configs/backend_operator_local_container_config.yaml
```

## 5. Running forecasting in the Data Science job

To run the forecasting operator within a Data Science job, follow these steps:

Adjust the `forecast.yaml` config with proper input/output folders. When the forecasting is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `forecast.yaml` and adjust the following fields:

```yaml
historical_data:
  url: oci://bucket@namespace/forecast/input_data/data.csv
model: prophet
output_directory:
  url: oci://bucket@namespace/forecast/result/
test_data:
  url: oci://bucket@namespace/forecast/input_data/test.csv
```

Publish the `forecast:v1` container to the [Oracle Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/home.htm). To become familiar with OCI, read the documentation links posted below.

- [Access Container Registry](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm#access)
- [Create repositories](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm#top)
- [Push images](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm#Pushing_Images_Using_the_Docker_CLI)

To publish `forecast:v1` to OCR, use the command posted below:

```bash
ads opctl operator publish-image forecast:v1 --registry <iad.ocir.io/tenancy/>
```

After the container is published to OCR, it can be used within Data Science jobs service. Check the `backend_job_container_config.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain an image property, something like `image: iad.ocir.io/<tenancy>/forecast:v1`. More details about supported options can be found in the ADS Jobs documentation - [Run a Container](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_container.html).

Run the forecasting on the Data Science jobs using the command posted below:

```bash
ads opctl apply -f /dst/configs/forecast.yaml --backend-config /dst/configs/backend_job_container_config.yaml
```

The logs can be monitored using the `ads opctl watch` command.
