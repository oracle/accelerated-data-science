# PII Operator



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
- `backend_operator_local_python_config.yaml`: This includes a local backend configuration for running pii in a local environment. The environment should be set up manually before running the operator.
- `backend_job_python_config.yaml`: Contains Data Science job-related config to run pii in a Data Science job within a conda runtime. The conda should be built and published before running the operator.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running PII on the local conda environment

To run forecasting locally, create and activate a new conda environment (`ads-pii`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- datapane
- scrubadub
- "git+https://github.com/oracle/accelerated-data-science.git@feature/forecasting#egg=oracle-ads"
```

Please review the previously generated `pii.yaml` file using the `init` command, and make any necessary adjustments to the input and output file locations. By default, it assumes that the files should be located in the same folder from which the `init` command was executed.

Use the command below to verify the pii config.

```bash
ads operator verify -f ~/pii/pii.yaml
```

Use the following command to run the forecasting within the `ads-pii` conda environment.

```bash
ads operator run -f ~/pii/pii.yaml -b local
```

The operator will run in your local environment without requiring any additional modifications.

## 4. Running PII in the Data Science job within conda runtime

To execute the forecasting operator within a Data Science job using conda runtime, please follow the steps outlined below:

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


After the conda environment is published to Object Storage, it can be used within Data Science jobs service. Check the `backend_job_python_config.yaml` config file. It should contain pre-populated infrastructure and runtime sections. The runtime section should contain a `conda` section.

```yaml
conda:
  type: published
  uri: oci://bucket@namespace/conda_environments/cpu/pii/1/pii_v1
```

More details about supported options can be found in the ADS Jobs documentation - [Run a Python Workload](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/run_python.html).

Adjust the `pii.yaml` config with proper input/output folders. When the pii is run in the Data Science job, it will not have access to local folders. Therefore, input data and output folders should be placed in the Object Storage bucket. Open the `pii.yaml` and adjust the following fields:

```yaml
output_directory:
  url: oci://bucket@namespace/pii/result/
test_data:
  url: oci://bucket@namespace/pii/input_data/test.csv
```

Run the pii on the Data Science jobs using the command posted below:

```bash
ads operator run -f ~/pii/pii.yaml --backend-config ~/pii/backend_job_python_config.yaml
```

The logs can be monitored using the `ads opctl watch` command.

```bash
ads opctl watch <OCID>
```
