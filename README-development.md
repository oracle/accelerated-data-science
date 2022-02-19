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
and set the Python version you want to use. ADS SDK requires Python >=3.7.

```bash
    conda create -n <yourenvname> python=3.7 anaconda
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

## Security

Consult the [security guide](./SECURITY.md) for our responsible security
vulnerability disclosure process.

## License

Copyright (c) 2020, 2022 Oracle, Inc. All rights reserved.
Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl.
