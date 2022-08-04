.. _Jobs Dataflow:

Run a Data Flow Application
***************************

Oracle Cloud Infrastructure (OCI) `Data Flow <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_getting_started.htm>`__ is a service for creating and running Spark applications. The following examples demonstrate how to create and run Data Flow applications using ADS.

Python
======

To create and run a Data Flow application, you must specify a compartment and a bucket for storing logs under the same compartment:

.. code-block:: python3

    compartment_id = "<compartment_id>"
    logs_bucket_uri = "<logs_bucket_uri>"

Ensure that you set up the correct policies. For instance, for Data Flow to access logs bucket, use a policy like:

::

   ALLOW SERVICE dataflow TO READ objects IN tenancy WHERE target.bucket.name='dataflow-logs'

For more information, see the `Data Flow documentation <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_getting_started.htm#set_up_admin>`__.

Update ``oci_profile`` if you're not using the default:

.. code-block:: python3

    oci_profile = "DEFAULT"
    config_location = "~/.oci/config"
    ads.set_auth(auth="api_key", oci_config_location=config_location, profile=oci_profile)

To create a Data Flow application you need two components: 

* ``DataFlow``, a subclass of ``Infrastructure``.
* ``DataFlowRuntime``, a subclass of ``Runtime``.

``DataFlow`` stores properties specific to Data Flow service, such as compartment_id, logs_bucket_uri, and so on.  You can set them using the ``with_{property}`` functions:

* ``with_compartment_id``
* ``with_configuration``
* ``with_driver_shape``
* ``with_executor_shape``
* ``with_language``
* ``with_logs_bucket_uri``
* ``with_metastore_id`` (`doc <https://docs.oracle.com/en-us/iaas/data-flow/using/hive-metastore.htm>`__)
* ``with_num_executors``
* ``with_spark_version``
* ``with_warehouse_bucket_uri``

For more details, see ```DataFlow`` class documentation <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/ads.jobs.html#module-ads.jobs.builders.infrastructure.dataflow>`__.

``DataFlowRuntime`` stores properties related to the script to be run, such as the path to the script and CLI arguments. Likewise all properties can be set using ``with_{property}``.  The ``DataFlowRuntime`` properties are:

* ``with_archive_bucket``
* ``with_archive_uri`` (`doc <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_data_flow_library.htm#third-party-libraries>`__)
* ``with_script_bucket``
* ``with_script_uri``

For more details, see the `runtime class documentation <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/ads.jobs.html#module-ads.jobs.builders.runtimes.python_runtime>`__.

Since service configurations remain mostly unchanged across multiple experiments, a ``DataFlow`` object can be reused and combined with various ``DataFlowRuntime`` parameters to create applications.

In the following "hello-world" example, ``DataFlow`` is populated with ``compartment_id``, ``driver_shape``, ``executor_shape``, and ``spark_version``.  ``DataFlowRuntime`` is populated with ``script_uri`` and ``script_bucket``. The ``script_uri`` specifies the path to the script. It can be local or remote (an Object Storage path). If the path is local, then ``script_bucket`` must be specified additionally because Data Flow requires a script to be available in Object Storage. ADS performs the upload step for you, as long as you give the bucket name or the Object Storage path prefix to upload the script. Either can be given to ``script_bucket``. For example,  either ``with_script_bucket("<bucket_name>")`` or ``with_script_bucket("oci://<bucket_name>@<namespace>/<prefix>")`` is accepted. In the next example, the prefix is given for ``script_bucket``.

.. code-block:: python3
    
    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime 
    from uuid import uuid4

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "script.py"), "w") as f:
            f.write('''
    import pyspark

    def main():
        print("Hello World")
        print("Spark version is", pyspark.__version__)

    if __name__ == "__main__":
        main()
            ''')
        name = f"dataflow-app-{str(uuid4())}"
        dataflow_configs = DataFlow()\
            .with_compartment_id(compartment_id)\
            .with_logs_bucket_uri(logs_bucket_uri)\
            .with_driver_shape("VM.Standard2.1") \
            .with_executor_shape("VM.Standard2.1") \
            .with_spark_version("3.2.1")
        runtime_config = DataFlowRuntime()\
            .with_script_uri(os.path.join(td, "script.py"))\
            .with_script_bucket(script_prefix)
        df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
        df.create()

To run this application, you could use:

.. code-block:: python3

    df_run = df.run()

After the run completes, check the ``stdout`` log from the application by running:

.. code-block:: python3

    print(df_run.logs.application.stdout)

You should this in the log:

.. code-block:: python3
    
    Hello World
    Spark version is 3.2.1

Data Flow supports adding third-party libraries using a ZIP file, usually called ``archive.zip``, see the `Data Flow documentation <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_data_flow_library.htm#third-party-libraries>`__ about how to create ZIP files. Similar to scripts, you can specify an archive ZIP for a Data Flow application using ``with_archive_uri``.  In the next example, ``archive_uri`` is given as an Object Storage location.  ``archive_uri`` can also be local so you must specify ``with_archive_bucket`` and follow the same rule as ``with_script_bucket``.

.. code-block:: python3
	
    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime 
    from uuid import uuid4

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "script.py"), "w") as f:
            f.write('''
    from pyspark.sql import SparkSession
    import click


    @click.command()
    @click.argument("app_name")
    @click.option(
        "--limit", "-l", help="max number of row to print", default=10, required=False
    )
    @click.option("--verbose", "-v", help="print out result in verbose mode", is_flag=True)
    def main(app_name, limit, verbose):
        # Create a Spark session
        spark = SparkSession.builder.appName(app_name).getOrCreate()

        # Load a csv file from dataflow public storage
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("multiLine", "true")
            .load(
                "oci://oow_2019_dataflow_lab@bigdatadatasciencelarge/usercontent/kaggle_berlin_airbnb_listings_summary.csv"
            )
        )

        # Create a temp view and do some SQL operations
        df.createOrReplaceTempView("berlin")
        query_result_df = spark.sql(
            """
            SELECT
                city,
                zipcode,
                CONCAT(latitude,',', longitude) AS lat_long
            FROM berlin
        """
        ).limit(limit)

        # Convert the filtered Spark DataFrame into JSON format
        # Note: we are writing to the spark stdout log so that we can retrieve the log later at the end of the notebook.
        if verbose:
            rows = query_result_df.toJSON().collect()
            for i, row in enumerate(rows):
                print(f"record {i}")
                print(row)


    if __name__ == "__main__":
        main()
            ''')

        name = f"dataflow-app-{str(uuid4())}"
        dataflow_configs = DataFlow()\
            .with_compartment_id(compartment_id)\
            .with_logs_bucket_uri(logs_bucket_uri)\
            .with_driver_shape("VM.Standard2.1") \
            .with_executor_shape("VM.Standard2.1") \
            .with_spark_version("3.2.1")
        runtime_config = DataFlowRuntime()\
            .with_script_uri(os.path.join(td, "script.py"))\
            .with_script_bucket("oci://<bucket>@<namespace>/prefix/path") \
            .with_archive_uri("oci://<bucket>@<namespace>/prefix/archive.zip")
        df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
        df.create()

You can pass arguments to a Data Flow run as a list of strings:

.. code-block:: python3

    df_run = df.run(args=["run-test", "-v", "-l", "5"])

You can save the application specification into a YAML file for future reuse. You could also use the ``json`` format.

.. code-block:: python3

    print(df.to_yaml("sample-df.yaml"))

You can also load a Data Flow application directly from the YAML file saved in the previous example:

.. code-block:: python3

    df2 = Job.from_yaml(uri="sample-df.yaml")

Create a new job and a run:

.. code-block:: python3

    df_run2 = df2.create().run()

Deleting a job cancels associated runs:

.. code-block:: python3

    df2.delete()
    df_run2.status

You can also load a Data Flow application from an OCID:

.. code-block:: python3

    df3 = Job.from_dataflow_job(df.id)

Creating a run under the same application:

.. code-block:: python3

    df_run3 = df3.run()

Now, there are 2 runs under the ``df`` application:

.. code-block:: python3

    assert len(df.run_list()) == 2

When you run a Data Flow application, a ``DataFlowRun`` object is created.  You can check the status, wait for a run to finish, check its logs afterwards, or cancel a run in progress. For example:

.. code-block:: python3

    df_run.status
    df_run.wait()

Note that ``watch`` is an alias of ``wait``, so you can also call ``df_run.watch()``.

There are three types of logs for a run: 

* application log 
* driver log 
* executor log 

Each log consists of ``stdout`` and ``stderr``. For example, to access ``stdout`` from application log, you could use:

.. code-block:: python3

    df_run.logs.application.stdout

Then you could check it with:

::

   df_run.logs.application.stderr
   df_run.logs.executor.stdout
   df_run.logs.executor.stderr

You can also examine ``head`` or ``tail`` of the log, or download it to a local path. For example,

.. code-block:: python3

    log = df_run.logs.application.stdout
    log.head(n=1)
    log.tail(n=1)
    log.download(<local-path>)

For the sample script, the log prints first five rows of a sample dataframe in JSON and it looks like:

.. code-block:: python3
    
    record 0
    {"city":"Berlin","zipcode":"10119","lat_long":"52.53453732241747,13.402556926822387"}
    record 1
    {"city":"Berlin","zipcode":"10437","lat_long":"52.54851279221664,13.404552826587466"}
    record 2
    {"city":"Berlin","zipcode":"10405","lat_long":"52.534996191586714,13.417578665333295"}
    record 3
    {"city":"Berlin","zipcode":"10777","lat_long":"52.498854933130026,13.34906453348717"}
    record 4
    {"city":"Berlin","zipcode":"10437","lat_long":"52.5431572633131,13.415091104515707"}

Calling ``log.head(n=1)`` returns this:

.. code-block:: python3
    
    'record 0'

Calling ``log.tail(n=1)`` returns this:

.. code-block:: python3

    {"city":"Berlin","zipcode":"10437","lat_long":"52.5431572633131,13.415091104515707"}


A link to run the page in the OCI Console is given using the ``run_details_link`` property:

.. code-block:: python3

    df_run.run_details_link

To list Data Flow applications, a compartment id must be given with any optional filtering criteria. For example, you can filter by name of the application:

.. code-block:: python3

    Job.dataflow_job(compartment_id=compartment_id, display_name=name)

YAML
====

You can create a Data Flow job directly from a YAML string. You can pass a YAML string into the ``Job.from_yaml()`` function to build a Data Flow job:

.. code-block:: yaml

  kind: job
  spec:
    id: <dataflow_app_ocid>
    infrastructure:
      kind: infrastructure
      spec:
        compartmentId: <compartment_id>
        driverShape: VM.Standard2.1
        executorShape: VM.Standard2.1
        id: <dataflow_app_ocid>
        language: PYTHON
        logsBucketUri: <logs_bucket_uri>
        numExecutors: 1
        sparkVersion: 3.2.1
      type: dataFlow
    name: dataflow_app_name
    runtime:
      kind: runtime
      spec:
        scriptBucket: bucket_name
        scriptPathURI: oci://<bucket_name>@<namespace>/<prefix>
      type: dataFlow

**Data Flow Infrastructure YAML Schema**

.. code-block:: yaml

    kind:
        allowed:
            - infrastructure
        required: true
        type: string
    spec:
        required: true
        type: dict
        schema:
            compartmentId:
                required: false
                type: string
            displayName:
                required: false
                type: string
            driverShape:
                required: false
                type: string
            executorShape:
                required: false
                type: string
            id:
                required: false
                type: string
            language:
                required: false
                type: string
            logsBucketUri:
                required: false
                type: string
            metastoreId:
                required: false
                type: string
            numExecutors:
                required: false
                type: integer
            sparkVersion:
                required: false
                type: string
    type:
        allowed:
            - dataFlow
        required: true
        type: string

**Data Flow Runtime YAML Schema**

.. code-block:: yaml

    kind:
        allowed:
            - runtime
        required: true
        type: string
    spec:
        required: true
        type: dict
        schema:
            archiveBucket:
                required: false
                type: string
            archiveUri:
                required: false
                type: string
            args:
                nullable: true
                required: false
                schema:
                    type: string
                type: list
            conda:
                nullable: false
                required: false
                type: dict
                schema:
                    uri:
                        required: true
                        type: string
                    region:
                        required: False
                        type: string
                    authType:
                        required: false
                        allowed:
                            - "resource_principal"
                            - "api_keys"
                            - "instance_principal"
                    type:
                        allowed:
                            - published
                        required: true
                        type: string
            env:
                type: list
                required: false
                schema:
                    type: dict
            freeformTags:
                required: false
                type: dict
            scriptBucket:
                required: false
                type: string
            scriptPathURI:
                required: false
                type: string
    type:
        allowed:
            - dataFlow
        required: true
        type: string

