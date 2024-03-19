# Feature Store Marketplace Operator
The Feature Store Marketplace Operator simplifies the process of deploying Feature Store stack from Marketplace to your existing OKE cluster.

Below are the steps to configure and run the Marketplace Operator:
## 1. Prerequisites
Follow the [CLI Configuration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/configure.html) steps from the ADS documentation.

## 2. Generating configs

To generate starter configs, run the command below. This will create a list of YAML configs and place them in the `output` folder.

```bash
ads operator init -t feature_store_marketplace --overwrite --output ~/marketplace
```

The most important files expected to be generated are:

- `feature_store_marketplace.yaml`: Contains feature store marketplace OKE related configuration.

All generated configurations should be ready to use without the need for any additional adjustments. However, they are provided as starter kit configurations that can be customized as needed.

## 3. Running feature store marketplace on the local environment

Use the following command to run the forecasting on local environment.

```bash
ads operator run -f ~/marketplace/feature_store_marketplace.yaml -b marketplace
```

The operator will run in your local environment without requiring any additional modifications.
