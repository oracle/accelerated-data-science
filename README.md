# Oracle Accelerated Data Science (ADS)

[![PyPI](https://img.shields.io/pypi/v/oracle-ads.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-ads/) [![Python](https://img.shields.io/pypi/pyversions/oracle-ads.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-ads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://github.com/ambv/black)


The [Oracle Accelerated Data Science (ADS) SDK](https://accelerated-data-science.readthedocs.io/en/latest/index.html) is maintained by the Oracle Cloud Infrastructure (OCI) [Data Science service](https://docs.oracle.com/en-us/iaas/data-science/using/data-science.htm) team. It speeds up common data science activities by providing tools that automate and simplify common data science tasks. Additionally, provides data scientists a friendly pythonic interface to OCI services. Some of the more notable services are OCI Data Science, Model Catalog, Model Deployment, Jobs, ML Pipelines, Data Flow, Object Storage, Vault, Big Data Service, Data Catalog, and the Autonomous Database. ADS gives you an interface to manage the life cycle of machine learning models, from data acquisition to model evaluation, interpretation, and model deployment.

With ADS you can:

- Read datasets from Oracle Object Storage, Oracle RDBMS (ATP/ADW/On-prem), AWS S3 and other sources into `Pandas dataframes`.
- Tune models using hyperparameter optimization with the `ADSTuner` tool.
- Generate detailed evaluation reports of your model candidates with the `ADSEvaluator` module.
- Save machine learning models to the [OCI Data Science Model Catalog](https://docs.oracle.com/en-us/iaas/data-science/using/models-about.htm).
- Deploy models as HTTP endpoints with [Model Deployment](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm).
- Launch distributed ETL, data processing, and model training jobs in Spark with [OCI Data Flow](https://docs.oracle.com/en-us/iaas/data-flow/using/home.htm).
- Train machine learning models in OCI Data Science [Jobs](https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm).
- Define and run an end-to-end machine learning orchestration covering all the steps of machine learning lifecycle in a repeatable, continuous [ML Pipelines](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/pipeline/overview.html#).
- Manage the life cycle of conda environments through the `ads conda` command line interface (CLI).

## Installation

You have various options when installing ADS.

### Installing the oracle-ads base package

```bash
  python3 -m pip install oracle-ads
```

### Installing OCI AI Operators

To use the AI Forecast Operator, install the "forecast" dependencies using the following command:

```bash
  python3 -m pip install 'oracle_ads[forecast]>=2.9.0'
```

### Installing extras libraries

To work with gradient boosting models, install the `boosted` module. This module includes XGBoost and LightGBM model classes.

```bash
  python3 -m pip install 'oracle-ads[boosted]'
```

For big data use cases using Oracle Big Data Service (BDS), install the `bds` module. It includes the following libraries, `ibis-framework[impala]`, `hdfs[kerberos]` and `sqlalchemy`.

```bash
  python3 -m pip install 'oracle-ads[bds]'
```

To work with a broad set of data formats (for example, Excel, Avro, etc.) install the `data` module. It includes the `fastavro`, `openpyxl`, `pandavro`, `asteval`, `datefinder`, `htmllistparse`, and `sqlalchemy` libraries.

```bash
  python3 -m pip install 'oracle-ads[data]'
```

To work with geospatial data install the `geo` module. It includes the `geopandas` and libraries from the `viz` module.

```bash
  python3 -m pip install 'oracle-ads[geo]'
```

Install the `notebook` module to use ADS within a OCI Data Science service [notebook session](https://docs.oracle.com/en-us/iaas/data-science/using/manage-notebook-sessions.htm). This module installs `ipywidgets` and `ipython` libraries.

```bash
  python3 -m pip install 'oracle-ads[notebook]'
```

To work with ONNX-compatible run times and libraries designed to maximize performance and model portability, install the `onnx` module. It includes the following libraries, `onnx`, `onnxruntime`, `onnxmltools`, `skl2onnx`, `xgboost`, `lightgbm` and libraries from the `viz` module.

```bash
  python3 -m pip install 'oracle-ads[onnx]'
```

For infrastructure tasks, install the `opctl` module. It includes the following libraries, `oci-cli`, `docker`, `conda-pack`, `nbconvert`, `nbformat`, and `inflection`.

```bash
  python3 -m pip install 'oracle-ads[opctl]'
```

For hyperparameter optimization tasks install the `optuna` module. It includes the `optuna` and libraries from the `viz` module.

```bash
  python3 -m pip install 'oracle-ads[optuna]'
```

Install the `tensorflow` module to include `tensorflow` and libraries from the `viz` module.

```bash
  python3 -m pip install 'oracle-ads[tensorflow]'
```

For text related tasks, install the `text` module. This will include the `wordcloud`, `spacy` libraries.

```bash
  python3 -m pip install 'oracle-ads[text]'
```

Install the `torch` module to include `pytorch` and libraries from the `viz` module.

```bash
  python3 -m pip install 'oracle-ads[torch]'
```

Install the `viz` module to include libraries for visualization tasks. Some of the key packages are `bokeh`, `folium`, `seaborn` and related packages.

```bash
  python3 -m pip install 'oracle-ads[viz]'
```

See `pyproject.toml` file `[project.optional-dependencies]` section for full list of modules and its list of extra libraries.

**Note**

Multiple extra dependencies can be installed together. For example:

```bash
  python3 -m pip install  'oracle-ads[notebook,viz,text]'
```

## Documentation

  - [Oracle Accelerated Data Science SDK (ADS) Documentation](https://accelerated-data-science.readthedocs.io/en/latest/index.html)
  - [OCI Data Science and AI services Examples](https://github.com/oracle/oci-data-science-ai-samples)
  - [Oracle AI & Data Science Blog](https://blogs.oracle.com/ai-and-datascience/)
  - [OCI Documentation](https://docs.oracle.com/en-us/iaas/data-science/using/data-science.htm)

## Examples

### Load data from Object Storage

```python
  import ads
  from ads.common.auth import default_signer
  import oci
  import pandas as pd

  ads.set_auth(auth="api_key", oci_config_location=oci.config.DEFAULT_LOCATION, profile="DEFAULT")
  bucket_name = <bucket_name>
  key = <key>
  namespace = <namespace>
  df = pd.read_csv(f"oci://{bucket_name}@{namespace}/{key}", storage_options=default_signer())
```

### Load data from ADB

This example uses SQL injection safe binding variables.

```python
  import ads
  import pandas as pd

  connection_parameters = {
      "user_name": "<user_name>",
      "password": "<password>",
      "service_name": "<tns_name>",
      "wallet_location": "<file_path>",
  }

  df = pd.DataFrame.ads.read_sql(
      """
      SELECT *
      FROM SH.SALES
      WHERE ROWNUM <= :max_rows
      """,
      bind_variables={ max_rows : 100 },
      connection_parameters=connection_parameters,
  )
```

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)

Find Getting Started instructions for developers in [README-development.md](https://github.com/oracle/accelerated-data-science/blob/main/README-development.md)

## Security

Consult the security guide [SECURITY.md](https://github.com/oracle/accelerated-data-science/blob/main/SECURITY.md) for our responsible security vulnerability disclosure process.

## License

Copyright (c) 2020, 2024 Oracle and/or its affiliates. Licensed under the [Universal Permissive License v1.0](https://oss.oracle.com/licenses/upl/)
