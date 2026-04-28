# Regression Operator

The Regression Operator trains supervised tabular regression models through a YAML-based interface. The supported model values are `linear_regression`, `random_forest`, `knn`, `xgboost`, and `auto`.

When a run completes, the operator writes artifacts into `output_directory`. Depending on the configuration and available data, these can include `training_predictions.csv`, `test_predictions.csv`, `training_metrics.csv`, `test_metrics.csv`, `global_explanations.csv`, `report.html`, and `model.pkl`.

Below are the steps to configure and run the Regression Operator on different resources.

## 1. Prerequisites

Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation. This step is required if you want `ads operator init` to generate OCI Data Science job backend configs for the Regression Operator.

## 2. Generating configs

To generate starter configs, run the command below. This creates configuration files in the `output` folder.

```bash
ads operator init -t regression --overwrite --output ~/regression/
```

The generated files include:

- `regression.yaml`: Contains regression operator configuration.
- `regression_operator_local_python_backend.yaml`: Local python backend configuration.
- `regression_operator_local_container_backend.yaml`: Local container backend configuration.

If OCI CLI defaults are configured for ADS jobs, `init` also generates job backend configs:

- `regression_job_container_backend.yaml`: Data Science job config for container runtime.
- `regression_job_python_backend.yaml`: Data Science job config for conda/python runtime.

All of these are starter configs and can be customized as needed.

## 3. Running regression on the local conda environment

To run regression locally, create and activate a conda environment and install the libraries listed in `environment.yaml`.

```yaml
- oracle-ads>=2.9.0
- report-creator
- cerberus
- scikit-learn
- shap
- xgboost
```

Then review `regression.yaml` and fill in the required operator inputs. At minimum, the schema requires `spec.training_data` and `spec.target_column`. `spec.test_data` is optional.

Example:

```yaml
kind: operator
type: regression
version: v1
spec:
  training_data:
    url: /path/to/train.csv
  test_data:
    url: /path/to/test.csv
  output_directory:
    url: /path/to/results
  target_column: target
  model: random_forest
```

Use the command below to verify the regression config.

```bash
ads operator verify -f ~/regression/regression.yaml
```

Use the following command to run the Regression Operator locally.

```bash
ads operator run -f ~/regression/regression.yaml -b local
```

## 4. Running regression on the local container

To run the Regression Operator within a local container, follow these steps:

Use the command below to build the regression container.

```bash
ads operator build-image -t regression
```

This creates a `regression:v1` image.

Check the `regression_operator_local_container_backend.yaml` config file. The generated config includes the image reference and mounts the `.oci` folder by default:

```yaml
kind: operator.local
spec:
  env:
  - name: operator
    value: regression:v1
  image: regression:v1
  volume:
  - /Users/<user>/.oci:/root/.oci
type: container
version: v1
```

Mounting `.oci` is only required if the operator will read from or write to OCI Object Storage. If you want the container to use local CSV files, add volume mounts for those folders and update `regression.yaml` to point to the in-container paths.

Run the Regression Operator in the container using the command below:

```bash
ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_operator_local_container_backend.yaml
```

## 5. Running regression in a Data Science job within container runtime

To execute the Regression Operator within a Data Science job using container runtime, follow these steps:

Use the following command to build the regression container. You can skip this if you already built it for the local container flow.

```bash
ads operator build-image -t regression
```

Publish the image to Oracle Container Registry:

```bash
ads operator publish-image -t regression --registry <iad.ocir.io/tenancy/>
```

After the image is published, check `regression_job_container_backend.yaml`. Its runtime section should reference the published regression image.

When running in a Data Science job, the operator cannot access local folders. Update `regression.yaml` so the data and output paths point to Object Storage.

Example:

```yaml
spec:
  training_data:
    url: oci://bucket@namespace/regression/input_data/train.csv
  test_data:
    url: oci://bucket@namespace/regression/input_data/test.csv
  output_directory:
    url: oci://bucket@namespace/regression/result/
```

Run the operator on Data Science Jobs using:

```bash
ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_job_container_backend.yaml
```

The logs can be monitored using:

```bash
ads opctl watch <OCID>
```

## 6. Running regression in a Data Science job within conda runtime

To execute the Regression Operator within a Data Science job using conda runtime, follow these steps:

Build the regression conda pack:

```bash
ads operator build-conda -t regression
```

Publish the regression conda pack:

```bash
ads operator publish-conda -t regression
```

After the conda pack is published, check `regression_job_python_backend.yaml`. Its runtime section should contain the conda configuration used by the job runtime.

As with the container job flow, update `regression.yaml` so `training_data`, optional `test_data`, and `output_directory` point to Object Storage locations before submitting the job.

Run the operator on Data Science Jobs using:

```bash
ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_job_python_backend.yaml
```

The logs can be monitored using:

```bash
ads opctl watch <OCID>
```
