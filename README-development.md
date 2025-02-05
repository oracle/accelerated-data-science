<!-- TOC -->
# Summary

The Oracle Accelerated Data Science (ADS) SDK used by data scientists and analysts for
data exploration and experimental machine learning to democratize machine learning and
analytics by providing easy-to-use, 
performant, and user friendly tools that
brings together the best of data science practices.

The ADS SDK helps you connect to different data sources, perform exploratory data analysis,
data visualization, feature engineering, model training, model evaluation, and
model interpretation. ADS also allows you to connect to the model catalog to save and load
models to and from the catalog.

- [Summary](#summary)
  - [Documentation](#documentation)
  - [Get Support](#get-support)
  - [Getting started](#getting-started)
    - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
    - [Step 2: Activate your environment](#step-2-activate-your-environment)
    - [Step 3: Clone ADS and install dependencies](#step-3-clone-ads-and-install-dependencies)
    - [Step 4: Setup configuration files](#step-4-setup-configuration-files)
    - [Step 5: Versioning and generation the wheel](#step-5-versioning-and-generation-the-wheel)
  - [Running tests](#running-tests)
    - [Running default setup tests](#running-default-setup-tests)
    - [Running all unit tests](#running-all-unit-tests)
    - [Running integration tests](#running-integration-tests)
    - [Running opctl integration tests](#running-opctl-integration-tests)
  - [Local Setup of AQUA API JupyterLab Server](#local-setup-of-aqua-api-jupyterlab-server)
    - [Step 1: Requirements](#step-1-requirements)
    - [Step 2: Create local .env files](#step-2-create-local-env-files)
    - [Step 3: Add the run\_ads.sh script in the ADS Repository](#step-3-add-the-run_adssh-script-in-the-ads-repository)
    - [Step 4: Run the JupyterLab Server](#step-4-run-the-jupyterlab-server)
    - [Step 5: Run the unit tests for the AQUA API](#step-5-run-the-unit-tests-for-the-aqua-api)
  - [Security](#security)
  - [License](#license)


## Documentation

 - [ads-documentation](https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/index.html)
 - [oci-data-science-ai-samples](https://github.com/oracle/oci-data-science-ai-samples)

## Get Support

- Open a [GitHub issue](https://github.com/oracle/accelerated-data-science/issues) for bug reports, questions, or requests for enhancements.
- Report a security vulnerability according to the [Reporting Vulnerabilities guide](https://www.oracle.com/corporate/security-practices/assurance/vulnerability/reporting.html).

## Getting started

These are the minimum required steps to install and set up the ADS SDK to run on your local machine
for development and testing purposes.

### Step 1: Create a conda environment

Install Anaconda from `https://repo.continuum.io/miniconda/` for the operating system you are using.

In the terminal client, enter the following where <yourenvname> is the name you want to call your environment,
and set the Python version you want to use. ADS SDK requires Python >=3.8.

```bash
    conda create -n <yourenvname> python=3.8 anaconda
```

This installs the Python version and all the associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/<yourenvname>`

### Step 2: Activate your environment

To activate or switch into your conda environment, run this command:

```bash
    conda activate <yourenvname>
```

To list of all your environments, use the `conda env list` command.

### Step 3: Clone ADS and install dependencies

Open the destination folder where you want to clone ADS library, and install dependencies like this:

```bash
    cd <desctination_folder>
    git clone git@github.com:oracle/accelerated-data-science.git
    python3 -m pip install -e .
```

To view which packages were installed and their version numbers, run:

```bash
    python3 -m pip freeze
```

### Step 4: Setup configuration files

You should also set up configuration files, see the [SDK and CLI Configuration File](https://docs.cloud.oracle.com/Content/API/Concepts/sdkconfig.htm).


### Step 5: Versioning and generation the wheel

Bump the versions in `pyproject.toml`. The ADS SDK using [build](https://pypa-build.readthedocs.io/en/stable/index.html) as build frontend. To generate sdist and wheel, you can run:

```bash
    pip install build
    python3 -m build
```

This wheel can then be installed using `pip`.

## Running tests

The SDK uses pytest as its test framework.

### Running default setup tests

Default setup tests for testing ADS SDK without extra dependencies, specified in `pyproject.toml` in `[project.optional-dependencies]`.

```bash
  # Update your environment with tests dependencies
  pip install -r test-requirements.txt
  # Run default setup tests
  python3 -m pytest tests/unitary/default_setup
```

### Running all unit tests

To run all unit test install extra dependencies to test all modules of ADS ASD.

```bash
  # Update your environment with tests dependencies
  pip install -r test-requirements.txt
  pip install -e ".[testsuite]"
  # Run all unit tests
  python3 -m pytest tests/unitary
```

### Running integration tests

ADS opctl integration tests can't be run together with all other integration tests, they require special setup.
To run all but opctl integration tests, you can run:

```bash
  # Update your environment with tests dependencies
  pip install -r test-requirements.txt
  pip install -e ".[testsuite]"
  # Run integration tests
  python3 -m pytest tests/integration --ignore=tests/integration/opctl
```

### Running opctl integration tests

ADS opctl integration tests utilize cpu, gpu jobs images and need dataexpl_p37_cpu_v2 and pyspark30_p37_cpu_v3 Data Science Environments be installed, see the [About Conda Environments](https://docs.oracle.com/en-us/iaas/data-science/using/conda_understand_environments.htm).
To build development container, see the [Build Development Container Image](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/opctl/localdev/jobs_container_image.html).

```bash
  # Update your environment with tests dependencies
  pip install -r test-requirements.txt
  pip install -e ".[opctl]"
  pip install oci oci-cli
  # Build cpu and gpu jobs images
  ads opctl build-image -d job-local
  ads opctl build-image -g -d job-local  
  # Run opclt integration tests
  python3 -m pytest tests/integration/opctl
```

## Local Setup of AQUA API JupyterLab Server
These are the steps to run the AQUA (AI Quick Actions) API Server for development and testing purposes. The source code for the AQUA API Server is [here](https://github.com/oracle/accelerated-data-science/tree/21ba00b95aef8581991fee6c7d558e2f2b1680ac/ads/aqua) within this repository.

### Step 1: Requirements
+ Complete the [Getting Started](#getting-started) Section above, create a conda environment with python >3.9 or 3.10
+ install any Rest API Client in your IDE (Thunder Client on [vscode](https://marketplace.visualstudio.com/items?itemName=rangav.vscode-thunder-client) or Postman) 
+ Activate the conda environment from the Getting Started Section and run

```
pip install -r test-requirements.txt
```

### Step 2: Create local .env files 
Running the local JuypterLab server requires setting OCI authentication, proxy, and OCI namespace parameters. Adapt this .env file with your specific OCI profile and OCIDs to set these variables.

```
CONDA_BUCKET_NS="your_conda_bucket"
http_proxy=""
https_proxy=""
HTTP_PROXY=""
HTTPS_PROXY=""
OCI_ODSC_SERVICE_ENDPOINT="your_service_endpoint"
AQUA_SERVICE_MODELS_BUCKET="service-managed-models"
AQUA_TELEMETRY_BUCKET_NS="" 
PROJECT_COMPARTMENT_OCID="ocid1.compartment.oc1.<your_ocid>" 
OCI_CONFIG_PROFILE="your_oci_profile_name"
OCI_IAM_TYPE="security_token" # no modification needed if using token-based auth
TENANCY_OCID="ocid1.tenancy.oc1.<your_ocid>"
AQUA_JOB_SUBNET_ID="ocid1.subnet.oc1.<your_ocid>"
ODSC_MODEL_COMPARTMENT_OCID="ocid1.compartment.oc1.<your_ocid>" 
PROJECT_OCID="ocid1.datascienceproject.oc1.<your_ocid>"
```

### Step 3: Add the run_ads.sh script in the ADS Repository 
+ add the shell script below and .env file from step 2 to your local directory of the cloned ADS Repository
+ Run ```chmox +x run_ads.sh``` after you create this script.
```
#!/bin/bash

#### Check if a CLI command is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <cli command>"
  exit 1
fi

#### Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env.int | xargs)
else
  echo "Error: .env.int file not found!"
  exit 1
fi

# Execute the CLI command
"$@"
```

### Step 4: Run the JupyterLab Server 
We can start the JupyterLab server using the following command

``` 
./run_ads.sh jupyter lab --no-browser --ServerApp.disable_check_xsrf=True
```
+ run ```pkill jupyter-lab``` to kill the JupyterLab server and re-run server to reflect changes made locally to the AQUA API
+ to test if server is running via CLI, run this in terminal

```
./run_ads.sh ads aqua model list
```

To make calls to the API, use the link http://localhost:8888/aqua/insert_handler_here with a REST API Client like Thunder Client/ Postman.

Examples of handlers
```
GET http://localhost:8888/aqua/model # calling the model_handler.py

GET http://localhost:8888/aqua/deployments # calling the deployment_handler.py
```
Handlers can be found [here](https://github.com/oracle/accelerated-data-science/tree/21ba00b95aef8581991fee6c7d558e2f2b1680ac/ads/aqua/extension).

### Step 5: Run the unit tests for the AQUA API
All the unit tests can be found [here](https://github.com/oracle/accelerated-data-science/tree/main/tests/unitary/with_extras/aqua). 
The following commands detail how the unit tests can be run.
```
# Run all tests in AQUA project
python -m pytest -q tests/unitary/with_extras/aqua/test_deployment.py

# Run all tests specific to a module within in AQUA project (ex. test_deployment.py, test_model.py, etc.)
python -m pytest -q tests/unitary/with_extras/aqua/test_deployment.py

# Run specific test method within the module (replace test_get_deployment_default_params with targeted test method)
python -m pytest tests/unitary/with_extras/aqua/test_deployment.py -k "test_get_deployment_default_params"
```

## Security

Consult the [security guide](./SECURITY.md) for our responsible security
vulnerability disclosure process.

## License

Copyright (c) 2020, 2022 Oracle, Inc. All rights reserved.
Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl.
