# PII Operator

The PII Operator ...

Below are the steps to configure and run the PII Operator on different resources.

## 1. Prerequisites

Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation. This step is mandatory as it sets up default values for different options while running the PII Operator on OCI Data Science jobs or OCI Data Flow applications. If you have previously done this and used a flexible shape, make sure to adjust `ml_job_config.ini` with shape config details and `docker_registry` information.

- ocpus = 1
- memory_in_gbs = 16
- docker_registry = `<iad.ocir.io/namespace/>`

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads operator init -t pii --overwrite --output ~/pii/
```

The most important files expected to be generated are:

- `pii.yaml`: Contains PII-related configuration.
- `backend_operator_local_python_config.yaml`: This includes a local backend configuration for running PII in a local environment. The environment should be set up manually before running the operator.
- `backend_operator_local_container_config.yaml`: This includes a local backend configuration for running PII within a local container. The container should be built before running the operator. Please refer to the instructions below for details on how to accomplish this.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running PII on the local conda environment

To run PII locally, create and activate a new conda environment (`ads-pii`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- "git+https://github.com/oracle/accelerated-data-science.git@feature/pii#egg=oracle-ads"
```

Please review the previously generated `pii.yaml` file using the `init` command, and make any necessary adjustments to the input and output file locations. By default, it assumes that the files should be located in the same folder from which the `init` command was executed.

Use the command below to verify the PII config.

```bash
ads operator verify -f ~/pii/pii.yaml
```

Use the following command to run the PII within the `ads-pii` conda environment.

```bash
ads operator run -f ~/pii/pii.yaml -b local
```

The operator will run in your local environment without requiring any additional modifications.

## 4. Running PII on the local container

To run the PII operator within a local container, follow these steps:

Use the command below to build the PII container.

```bash
ads operator build-image -t pii
```

This will create a new `pii:v1` image, with `/etc/operator` as the designated working directory within the container.


Check the `backend_operator_local_container_config.yaml` config file. By default, it should have a `volume` section with the `.oci` configs folder mounted.

```yaml
volume:
  - "/Users/<user>/.oci:/root/.oci"
```

Mounting the OCI configs folder is only required if an OCI Object Storage bucket will be used to store the input PII data or output PII result. The input/output folders can also be mounted to the container.

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
  image: PII:v1
  volume:
  - /Users/<user>/.oci:/root/.oci
  - /Users/<user>/pii/data:/etc/operator/data
  - /Users/<user>/pii/result:/etc/operator/result
type: container
version: v1
```

Run the PII within a container using the command below:

```bash
ads operator run -f ~/pii/pii.yaml --backend-config ~/pii/backend_operator_local_container_config.yaml
```
