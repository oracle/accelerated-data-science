Conda Environment
*****************

To work with BDS in a notebook session or job, you must have a conda environment that supports the BDS module in ADS along with support for PySpark.
This section demonstrates how to modify a PySpark Data Science conda environment to work with BDS. It also demonstrates how to publish this conda environment so that you can be share it with team members and use it in jobs.

Create
======

.. include:: _template/create_conda.rst

Publish
=======

* Create an Object Storage bucket to store published conda environments.
* Open a terminal window then run the following commands and actions:
* ``odsc conda init -b <bucket_name> -b <namespace> -a <resource_principal or api_key>``: Initialize the environment so that you can work with Published Conda Environments.
* ``odsc conda publish -s pyspark30_p37_cpu_v3``: Publish the conda environment.
* In the OCI Console, open Data Science.
* Select a project.
* Select a click the notebook session's name, or the Actions menu, and click Open to open the notebook session's JupyterLab interface in another tab..
* Click Published Conda Environments in the Environment Explorer tab to list all the published conda environments that are available in your designated Object Storage bucket.
* Select the Environment Version that you specified.
* Click the copy button adjacent to the Source conda environment to copy the file source path to use when installing the conda environment in other notebook sessions or to use with jobs.
