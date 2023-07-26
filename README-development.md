# Summary

The Oracle Accelerated Data Science (ADS) SDK used by data scientists and analysts for
data exploration and experimental machine learning to democratize machine learning and
analytics by providing easy-to-use, performant, and user friendly tools that
brings together the best of data science practices.

The ADS SDK helps you connect to different data sources, perform exploratory data analysis,
data visualization, feature engineering, model training, model evaluation, and
model interpretation. ADS also allows you to connect to the model catalog to save and load
models to and from the catalog.

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

To which packages were installed and their version numbers, run:

```bash
    python3 -m pip freeze
```

### Step 4: Setup configuration files

You should also set up configuration files, see the [SDK and CLI Configuration File](https://docs.cloud.oracle.com/Content/API/Concepts/sdkconfig.htm).


### Step 5: Versioning and generation the wheel

Use `ads_version.json` for versioning. The ADS SDK is packaged as a wheel. To generate the wheel, you can run:

```bash
    python3 setup.py sdist bdist_wheel
```

This wheel can then be installed using `pip`.

## Running tests

The SDK uses pytest as its test framework.

### Running default setup tests

Default setup tests for testing ADS SDK without extra dependencies, specified in setup.py.

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
  pip install -r dev-requirements.txt
  # Run all unit tests
  python3 -m pytest tests/unitary
```

### Running integration tests

ADS opctl integration tests can't be run together with all other integration tests, they require special setup.
To run all but opctl integration tests, you can run:

```bash
  # Update your environment with tests dependencies
  pip install -r dev-requirements.txt
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

## Security

Consult the [security guide](./SECURITY.md) for our responsible security
vulnerability disclosure process.

## License

Copyright (c) 2020, 2022 Oracle, Inc. All rights reserved.
Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl.
