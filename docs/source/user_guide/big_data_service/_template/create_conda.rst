The following are the recommended steps to create a conda environment to connect to BDS:

- Open a terminal window then run the following commands:
- ``odsc conda install -s pyspark30_p37_cpu_v3``: Install the PySpark conda environment.
- ``conda activate /home/datascience/conda/pyspark30_p37_cpu_v3``: Activate the PySpark conda environment so that you can modify it.
- ``pip uninstall oracle_ads``: Uninstall the old ADS package in this environment.
- ``pip install oracle_ads[bds]``: Install the latest version of ADS that contains BDS support.
- ``conda install sasl``: Install ``sasl``.

