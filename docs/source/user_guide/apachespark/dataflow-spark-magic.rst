####################
OCI Data Flow Studio
####################

This section demonstrates how to run interactive Spark workloads on a long lasting `Oracle Cloud Infrastructure Data Flow <https://docs.oracle.com/iaas/data-flow/using/home.htm>`__ cluster through `Apache Livy <https://livy.apache.org/>`__ integration.

**Data Flow Studio allows you to:**

* Run Spark code against a Data Flow remote Spark cluster
* Create a Data Flow Spark session with SparkContext and HiveContext against a Data Flow remote Spark cluster
* Capture the output of Spark queries as a local Pandas data frame to interact easily with other Python libraries (e.g. matplotlib)

**Key Features & Benefits:**

* Data Flow sessions support auto-scaling Data Flow cluster capabilities
* Data Flow sessions support the use of conda environments as customizable Spark runtime environments

**Limitations:**

* Data Flow sessions can last up to 7 days or 10,080 mins (maxDurationInMinutes).
* Data Flow Sessions can only be accessed through OCI Data Science Notebook Sessions.
* Not all SparkMagic commands are currently supported. To see the full list, run the ``%help`` command in a notebook cell.

**Notebook Examples:**

* `Introduction to the Oracle Cloud Infrastructure Data Flow Studio <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/master/notebook_examples/pyspark-data_flow_studio-introduction.ipynb>`__
* `Spark NLP within Oracle Cloud Infrastructure Data Flow Studio <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/master/notebook_examples/pyspark-data_flow_studio-spark_nlp.ipynb>`__

Prerequisite
============
Data Flow Sessions are accessible through the following conda environment:

* PySpark 3.2 and Data Flow 2.0 (pyspark32_p38_cpu_v2)

You can customize **pypspark32_p38_cpu_v1**, publish it, and use it as a runtime environment for a Data Flow Session.

Policies
********

Data Flow requires policies to be set in IAM to access resources to manage and run sessions. Refer to the `Data Flow Studio Policies <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_getting_started.htm#policies-data-flow-studio>`__ documentation on how to setup policies.


Quick Start
===========

.. code-block:: python

  import ads
  ads.set_auth("resource_principal")

  %load_ext dataflow.magics

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"}'

  %%spark
  print(sc.version)

Data Flow Spark Magic
=====================
Data Flow Spark Magic is used for interactively working with remote Spark clusters through Livy, a Spark REST server, in Jupyter notebooks. It is a JupyterLab extension that you need to activate in your notebook.

.. code-block:: python

  %load_ext dataflow.magics

Use the `%help` method to get a list of all the available commands, along with a list of their arguments and example calls.

.. code-block:: python

  %help

.. admonition:: Tip

  To access the docstrings of any magic command and figure out what arguments to provide, simply add ``?`` at the end of the command. For instance: ``%create_session?``

Create Session
**************

**Example command for Flex shapes**

To create a new Data Flow cluster session use the ``%create_session`` magic command.

.. code-block:: python

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"}'

**Example command for Spark dynamic allocation (aka auto-scaling)**

To help you save resources and reduce time on management, Spark `dynamic allocation <https://docs.oracle.com/iaas/data-flow/using/dynamic-alloc-about.htm#dynamic-alloc-about>`__ is now enabled in Data Flow. You can define a Data Flow cluster based on a range of executors, instead of just a fixed number of executors. Spark provides a mechanism to dynamically adjust the resources the application occupies based on the workload. The application might relinquish resources if they are no longer used and request them again later when there is demand.

.. code-block:: python

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"\
    "configuration":{\
      "spark.dynamicAllocation.enabled":"true",\
        "spark.dynamicAllocation.shuffleTracking.enabled":"true",\
        "spark.dynamicAllocation.minExecutors":"1",\
        "spark.dynamicAllocation.maxExecutors":"4",\
        "spark.dynamicAllocation.executorIdleTimeout":"60",\
        "spark.dynamicAllocation.schedulerBacklogTimeout":"60",\
        "spark.dataflow.dynamicAllocation.quotaPolicy":"min"}}'

**Example command with third-party libraries**

The Data Flow Sessions support `custom dependencies <https://docs.oracle.com/iaas/data-flow/using/third-party-libraries.htm>`__ in the form of Python wheels or virtual environments. You might want to make native code or other assets available within your Spark runtime. The dependencies can be attached by using the `archiveUri` attribute.

.. code-block:: python

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "archiveUri":"oci://<bucket_name>@<namespace>/<zip_archive>",\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"}'

**Example command with the Data Catalog Hive Metastore**

The `Data Catalog Hive Metastore <https://docs.oracle.com/iaas/data-catalog/using/metastore.htm>`__  provides schema definitions for objects in structured and unstructured data assets. Use the `metastoreId` to access the Data Catalog Metastore.

.. code-block:: python

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "metastoreId": "<ocid1.datacatalogmetastore...>",\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"}'

**Example command with the published conda environment**

You can use a published conda environment as a Data Flow runtime environment.

* `Creating a Custom Conda Environment <https://docs.oracle.com/iaas/data-science/using/conda_create_conda_env.htm>`__
* `How to create a new conda environment in OCI Data Science <https://blogs.oracle.com/ai-and-datascience/post/creating-a-new-conda-environment-from-scratch-in-oci-data-science>`__
* `Publishing a Conda Environment to an Object Storage Bucket in Your Tenancy <https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm#:~:text=You%20can%20publish%20a%20conda%20environment%20that%20you%20have%20installed,persist%20them%20across%20notebook%20sessions.>`__

The path to the published conda environment can be copied from the `Environment Explorer <https://docs.oracle.com/iaas/data-science/using/conda_viewing.htm>`__.

Example path : ``oci://<your-bucket>@<your-tenancy-namespace>/conda_environments/cpu/PySpark 3.2 and Data Flow/2.0/pyspark32_p38_cpu_v2#conda``

.. code-block:: python

  %create_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "sparkVersion":"3.2.1",\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":1,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "logsBucketUri" : "oci://<bucket_name>@<namespace>/"\
    "configuration":{\
      "spark.archives": "oci://<your-bucket>@<your-tenancy-namespace>/conda_environments/cpu/PySpark 3.2 and Data Flow/2.0/pyspark32_p38_cpu_v2#conda>"}}'


Update Session
**************

You can modify the configuration of your running session using the ``%update_session`` command. For example, Data Flow sessions can last up to 7 days or 10080 mins (168 hours) (**maxDurationInMinutes**) and have default idle timeout value of 480 mins (8 hours)(**idleTimeoutInMinutes**). Only those two can be updated on a running cluster without re-creating the cluster.

.. code-block:: python

  %update_session -i '{"maxDurationInMinutes": 1440, "idleTimeoutInMinutes": 420}'

Configure Session
*****************

The existing session can be reconfigured with the ``%configure_session`` command. The new configuration will be applied the next time the session is started. Use the force flag ``-f`` to immediately drop and recreate the running cluster session.

.. code-block:: python

  %configure_session -f -i '{\
    "driverShape":"VM.Standard.E4.Flex",\
    "executorShape":"VM.Standard.E4.Flex",\
    "numExecutors":2,\
    "driverShapeConfig":{"ocpus":1,"memoryInGBs":16},\
    "executorShapeConfig":{"ocpus":1,"memoryInGBs":16}}'

Stop Session
************
To stop the current session, use the ``%stop_session`` magic command. You don't need to provide any arguments for this command. The current active cluster will be stopped. All data in memory will be lost.

.. code-block:: python

  %stop_session

Activate Session
****************
To re-activate the existing session, use the ``%activate_session`` magic command. The ``application_id`` can be taken from the console UI.

.. code-block:: python

  %activate_session -l python -c '{\
    "compartmentId":"<compartment_id>",\
    "displayName":"TestDataFlowSession",\
    "applicationId":"<application_id>"}'

Use Existing Session
********************
To connect to the existing session use the `%use_session` magic command.

.. code-block:: python

  %use_session -s "<application_id>"


Basic Spark Usage Examples
==========================
A SparkContext (``sc``) and HiveContext (``sqlContext``) are automatically created in the session cluster. The magic commands include the ``%%spark`` command to run Spark commands in the cluster. You can access information about the Spark application, define a dataframe where results are to be stored, modify the configuration, and so on.

The ``%%spark`` magic command comes with a number of parameters that allow you to interact with the Data Flow Spark cluster. Any cell content that starts with the ``%%spark`` command will be executed in the remote Spark cluster.

Check the Spark context version:

.. code-block:: python

  %%spark
  print(sc.version)


A toy example of how to use ``sc`` in a Data Flow Spark Magic cell:

.. code-block:: python

  %%spark
  numbers = sc.parallelize([4, 3, 2, 1])
  print(f"First element of numbers is {numbers.first()}")
  print(f"The RDD, numbers, has the following description\n{numbers.toDebugString()}")

Spark SQL
*********
Using the ``-c sql`` option allows you to run Spark SQL commands in a cell. In this section, the `NYC Taxi and Limousine Commission (TLC) Data <https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page>`__ dataset is used. The size of the dataset is around **35GB**.

The next cell reads the dataset into a Spark dataframe, and then saves it as a view used to demonstrate Spark SQL.


Use the ``-c sql`` option to run Spark SQL commands in a cell.

The next example demonstrates how a dataset can be created on the fly:

.. code-block:: python

  %%spark
  df_nyc_tlc = spark.read.parquet("oci://hosted-ds-datasets@bigdatadatasciencelarge/nyc_tlc/201[1,2,3,4,5,6,7,8]/**/data.parquet", header=False, inferSchema=True)
  df_nyc_tlc.show()
  df_nyc_tlc.createOrReplaceTempView("nyc_tlc")

The following cell uses the ``-c sql`` option to tell Data Flow Spark Magic that the contents of the cell is SparkSQL. The ``-o <variable>`` option takes the results of the Spark SQL operation and stores it in the defined variable. In this case, the ``df_people`` will be a Pandas dataframe that is available to be used in the notebook.

.. code-block:: python

  %%spark -c sql -o df_nyc_tlc
  SELECT vendor_id, passenger_count, trip_distance, payment_type FROM nyc_tlc LIMIT 1000;

Check the result:

.. code-block:: python

  print(type(df_nyc_tlc))
  df_nyc_tlc.head()