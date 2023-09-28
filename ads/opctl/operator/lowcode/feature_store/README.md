# Feature Store Operator

Managing many datasets, data-sources and transformations for machine learning is complex and costly. Poorly cleaned data, data issues, bugs in transformations, data drift and training serving skew all leads to increased model development time and worse model performance. Here, feature store is well positioned to solve many of the problems since it provides a centralised way to transform and access data for training and serving time and helps defines a standardised pipeline for ingestion of data and querying of data.

Below are the steps to configure and deploy the Feature Store on OCI resources.

## 1. Prerequisites

### User policies for stack setup
Feature Store users need to provide the following access permissions in order to deploy the feature store terraform stack. Below mentioned are the policy statements required for terraform stack deployment

 - define tenancy service_tenancy as <YOUR_OCID>
 - endorse group <feature store user group> to read repos in tenancy service_tenancy
 - allow group <feature store user group> to manage orm-stacks in compartment <compartmentName>
 - allow group <feature store user group> to manage orm-jobs in compartment <compartmentName>
 - allow group <feature store user group> to manage object-family in compartment <compartmentName>
 - allow group <feature store user group> to manage users in compartment <compartmentName>
 - allow group <feature store user group> to manage instance-family in compartment <compartmentName>
 - allow group <feature store user group> to manage tag-namespaces in compartment <compartmentName>
 - allow group <feature store user group> to manage groups in compartment <compartmentName>
 - allow group <feature store user group> to manage policies in compartment <compartmentName>
 - allow group <feature store user group> to manage dynamic-groups in compartment <compartmentName>
 - allow group <feature store user group> to manage virtual-network-family in compartment <compartmentName>
 - allow group <feature store user group> to manage functions-family in compartment <compartmentName>
 - allow group <feature store user group> to inspect compartments in compartment <compartmentName>
 - allow group <feature store user group> to manage cluster-family in compartment <compartmentName>
 - allow group <feature store user group> to manage mysql-family in compartment <compartmentName>
 - allow group <feature store user group> to manage api-gateway-family in compartment <compartmentName>

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads opctl operator init -n feature_store --overwrite --output ~/feature_store/
```

The most important files expected to be generated are:

- `feature_store.yaml`: Contains feature store related configuration.
- `backend_operator_local_python_config.yaml`: This includes a local backend configuration for running the operator in a local environment. The environment should be set up manually before running the operator.
- `backend_operator_local_container_config.yaml`: This includes a local backend configuration for running operator within a local container. The container should be built before running the operator. Please refer to the instructions below for details on how to accomplish this.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running the operator on the local environment

To run operator locally, create and activate a new conda environment (`ads-feature-store`). Install all the required libraries listed in the `environment.yaml` file.

```yaml
- oci-cli
- "git+https://github.com/oracle/accelerated-data-science.git@feature/feature_store_operator#egg=oracle-ads"
```

Please review the previously generated `feature_store.yaml` file using the `init` command, and make any necessary adjustments to the input and output file locations. By default, it assumes that the files should be located in the same folder from which the `init` command was executed.

Use the command below to verify the operator's config.

```bash
ads opctl operator verify -f ~/feature_store/feature_store.yaml
```

Use the following command to run the operator within the `ads-feature-store` conda environment.

```bash
ads opctl apply -f ~/feature_store/feature_store.yaml -b local
```

The operator will run in your local environment without requiring any additional modifications.

## 4. Running the operator on the local container

To run the the operator within a local container, follow these steps:

Use the command below to build the operator's container.

```bash
ads opctl operator build-image -n feature_store
```

This will create a new `feature_store:v1` image, with `/etc/operator` as the designated working directory within the container.


Check the `backend_operator_local_container_config.yaml` config file. By default, it should have a `volume` section with the `.oci` configs folder mounted.

```yaml
volume:
  - "/Users/<user>/.oci:/root/.oci"
```

Mounting the OCI configs folder is required because the container will need to access the OCI resources.

```yaml
volume:
  - /Users/<user>/.oci:/root/.oci
```

The full config can look like:
```yaml
kind: operator.local
spec:
  image: feature_store:v1
  volume:
  - /Users/<user>/.oci:/root/.oci
type: container
version: v1
```

Run the operator within a container using the command below:

```bash
ads opctl apply -f ~/feature_store/feature_store.yaml --backend-config ~/feature_store/backend_operator_local_container_config.yaml
```
