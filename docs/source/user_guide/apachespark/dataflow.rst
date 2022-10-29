===============================================
Running your Spark Application on OCI Data Flow
===============================================

Submit your code to DataFlow for workloads that require larger resources.

Notebook Extension
==================


For most Notebook users, local or OCI Notebook Sessions, the notebook extension is the most straightforward integration with dataflow. It's a "Set It and Forget It" API with options to update ad-hoc. You can configure your dataflow runs by running ``ads opctl configure`` in the terminal.

After setting up your dataflow config, you can return to the Notebook. Import ``ads`` and ``DataFlowConfig``:

.. code-block:: python

  import ads
  from ads.jobs.utils import DataFlowConfig

Load the dataflow extension inside the notebook cell -

.. code-block:: python

  %load_ext ads.jobs.extension

Define config. If you have not yet configured your dataflow setting, or would like to amend the defaults, you can modify as shown below:

.. code-block:: python

  dataflow_config = DataFlowConfig()
  dataflow_config.compartment_id = "ocid1.compartment.<your compartment ocid>"
  dataflow_config.driver_shape = "VM.Standard.E4.Flex"
  dataflow_config.driver_shape_config = oci.data_flow.models.ShapeConfig(ocpus=2, memory_in_gbs=32)
  dataflow_config.executor_shape = "VM.Standard.E4.Flex"
  dataflow_config.executor_shape_config = oci.data_flow.models.ShapeConfig(ocpus=4, memory_in_gbs=64)
  dataflow_config.logs_bucket_uri = "oci://<my-bucket>@<my-tenancy>/"
  dataflow_config.spark_version = "3.2.1"
  dataflow_config.configuration = {"spark.driver.memory": "512m"}

Use the config defined above to submit the cell.

.. admonition:: Tip

  Get more information about the dataflow extension by running ``%dataflow -h``

Call the dataflow magic command in the first line of your cell to run it on dataflow.

.. code-block:: python

    %%dataflow run -f a_script.py -c {dataflow_config} -w -o -- abc -l 5 -v


This header will:
- save the cell as a file called ``script.py`` and store it in your ``dataflow_config.script_bucket``
- After the ``--`` notation, all parameters are sent to your script. For example ``abc`` is a positional argument, and ``l`` and ``v`` are named arguments.


Below is a full example:

.. code-block:: python

    %%dataflow run -f a_script.py -c {dataflow_config} -w -o -- abc -l 5 -v
    from pyspark.sql import SparkSession
    import click


    @click.command()
    @click.argument("app_name")
    @click.option(
        "--limit", "-l", help="max number of rows to print", default=10, required=False
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


ADS CLI
=======

**Prerequisites**

1. :doc:`Install ADS CLI<../cli/quickstart>`
2. :doc:`Configure Defaults<../cli/opctl/configure>`

.. admonition:: Tip

    If, for some reason, you are unable to use CLI, instead skip to the ``Create, Run Data Flow Application Using ADS Python SDK`` section below.

Sometimes your code is too complex to run in a single cell, and it's better run as a notebook or file. In that case, use the ADS Opctl CLI.

To submit your notebook to DataFlow using the ``ads`` CLI, run:

.. code-block:: shell

  ads opctl run -s <folder where notebook is located> -e <notebook name> -b dataflow

.. admonition:: Tip

  You can avoid running cells that are not DataFlow environment compatible by tagging the cells and then providing the tag names to ignore. In the following example cells that are tagged ``ignore`` and ``remove`` will be ignored -
  ``--exclude-tag ignore --exclude-tag remove``

.. admonition:: Tip

  You can run the notebook in your local pyspark environment before submitting to ``DataFlow`` using the same CLI with ``-b local``

  .. code-block:: shell

    # Activate the Pyspark conda environment in local
    ads opctl run -s <notebook directory> -e <notebook file> -b local

You could submit a notebook using ADS SDK APIs. Here is an example to submit a notebook -

.. code-block:: python

    from ads.jobs import Job, DataFlow, DataFlowNotebookRuntime

    df = (
        DataFlow()
        .with_compartment_id(
            "ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq"
        )
        .with_driver_shape("VM.Standard.E4.Flex")
		.with_driver_shape_config(ocpus=2, memory_in_gbs=32)
		.with_executor_shape("VM.Standard.E4.Flex")
		.with_executor_shape_config(ocpus=4, memory_in_gbs=64)
        .with_logs_bucket_uri("oci://mybucket@mytenancy/")
    )
    rt = (
        DataFlowNotebookRuntime()
        .with_notebook(
            "<path to notebook>"
        )  # This could be local path or http path to notebook ipynb file
        .with_script_bucket("<my-bucket>")
        .with_exclude_tag(["ignore", "remove"])  # Cells to Ignore
    )
    job = Job(infrastructure=df, runtime=rt).create(overwrite=True)
    df_run = job.run(wait=True)



ADS Python SDK
==============

To create a Data Flow application using the ADS Python API you need two components:

- ``DataFlow``, a subclass of ``Infrastructure``.
- ``DataFlowRuntime``, a subclass of ``Runtime``.

``DataFlow`` stores properties specific to Data Flow service, such as
compartment_id, logs_bucket_uri, and so on.
You can set them using the ``with_{property}`` functions:

- ``with_compartment_id``
- ``with_configuration``
- ``with_driver_shape``
- ``with_driver_shape_config``
- ``with_executor_shape``
- ``with_executor_shape_config``
- ``with_language``
- ``with_logs_bucket_uri``
- ``with_metastore_id`` (`doc <https://docs.oracle.com/en-us/iaas/data-flow/using/hive-metastore.htm>`__)
- ``with_num_executors``
- ``with_spark_version``
- ``with_warehouse_bucket_uri``

For more details, see `DataFlow class documentation <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/ads.jobs.html#module-ads.jobs.builders.infrastructure.dataflow>`__.

``DataFlowRuntime`` stores properties related to the script to be run, such as the path to the script and
CLI arguments. Likewise all properties can be set using ``with_{property}``.
The ``DataFlowRuntime`` properties are:

- ``with_script_uri``
- ``with_script_bucket``
- ``with_archive_uri`` (`doc <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_data_flow_library.htm#third-party-libraries>`__)
- ``with_archive_bucket``
- ``with_custom_conda``

For more details, see the `runtime class documentation <../../ads.jobs.html#module-ads.jobs.builders.runtimes.python_runtime>`__.

Since service configurations remain mostly unchanged across multiple experiments, a ``DataFlow``
object can be reused and combined with various ``DataFlowRuntime`` parameters to
create applications.

In the following "hello-world" example, ``DataFlow`` is populated with ``compartment_id``,
``driver_shape``, ``driver_shape_config``, ``executor_shape``, ``executor_shape_config`` 
and ``spark_version``. ``DataFlowRuntime`` is populated with ``script_uri`` and
``script_bucket``. The ``script_uri`` specifies the path to the script. It can be
local or remote (an Object Storage path). If the path is local, then
``script_bucket`` must be specified additionally because Data Flow
requires a script to be available in Object Storage. ADS
performs the upload step for you, as long as you give the bucket name
or the Object Storage path prefix to upload the script. Either can be
given to ``script_bucket``. For example,  either
``with_script_bucket("<bucket_name>")`` or
``with_script_bucket("oci://<bucket_name>@<namespace>/<prefix>")`` is
accepted. In the next example, the prefix is given for ``script_bucket``.

.. code-block:: python

    from ads.jobs import DataFlow, Job, DataFlowRuntime
    from uuid import uuid4
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "script.py"), "w") as f:
            f.write(
                """
    import pyspark

    def main():
        print("Hello World")
        print("Spark version is", pyspark.__version__)

    if __name__ == "__main__":
        main()
            """
            )
        name = f"dataflow-app-{str(uuid4())}"
        dataflow_configs = (
            DataFlow()
            .with_compartment_id("oci.xx.<compartment_id>")
            .with_logs_bucket_uri("oci://mybucket@mynamespace/dflogs")
            .with_driver_shape("VM.Standard.E4.Flex")
		    .with_driver_shape_config(ocpus=2, memory_in_gbs=32)
		    .with_executor_shape("VM.Standard.E4.Flex")
		    .with_executor_shape_config(ocpus=4, memory_in_gbs=64)
            .with_spark_version("3.0.2")
        )
        runtime_config = (
            DataFlowRuntime()
            .with_script_uri(os.path.join(td, "script.py"))
            .with_script_bucket("oci://mybucket@namespace/prefix")
            .with_custom_conda("oci://<mybucket>@<mynamespace>/<path/to/conda_pack>")
        )
        df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
        df.create()


To run this application, you could use:

.. code-block:: python

    df_run = df.run()

After the run completes, check the ``stdout`` log from the application by running:

.. code-block:: python

    print(df_run.logs.application.stdout)

You should this in the log:

.. code-block:: python

    Hello World
    Spark version is 3.0.2

**Note on Policy**

.. parsed-literal::

   ALLOW SERVICE dataflow TO READ objects IN tenancy WHERE target.bucket.name='dataflow-logs'


Data Flow supports adding third-party libraries using a ZIP file, usually called ``archive.zip``, see the `Data Flow documentation <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_data_flow_library.htm#third-party-libraries>`__
about how to create ZIP files. Similar to scripts, you can specify an archive ZIP for a Data Flow application using ``with_archive_uri``.
In the next example, ``archive_uri`` is given as an Object Storage location.
``archive_uri`` can also be local so you must specify ``with_archive_bucket`` and follow the same rule as ``with_script_bucket``.

.. code-block:: python

    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime, Job
    from uuid import uuid4
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "script.py"), "w") as f:
            f.write(
                '''
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
            '''
            )

        name = f"dataflow-app-{str(uuid4())}"
        dataflow_configs = (
            DataFlow()
            .with_compartment_id("oci1.xxx.<compartment_ocid>")
            .with_logs_bucket_uri("oci://mybucket@mynamespace/prefix")
            .with_driver_shape("VM.Standard.E4.Flex")
		    .with_driver_shape_config(ocpus=2, memory_in_gbs=32)
		    .with_executor_shape("VM.Standard.E4.Flex")
		    .with_executor_shape_config(ocpus=4, memory_in_gbs=64)
            .with_spark_version("3.0.2")
        )
        runtime_config = (
            DataFlowRuntime()
            .with_script_uri(os.path.join(td, "script.py"))
            .with_script_bucket("oci://<bucket>@<namespace>/prefix/path")
            .with_archive_uri("oci://<bucket>@<namespace>/prefix/archive.zip")
            .with_custom_conda(uri="oci://<mybucket>@<mynamespace>/<my-conda-uri>")
        )
        df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
        df.create()


You can pass arguments to a Data Flow run as a list of strings:

.. code-block:: python

    df_run = df.run(args=["run-test", "-v", "-l", "5"])

You can save the application specification into a YAML file for future
reuse. You could also use the ``json`` format.

.. code-block:: python

    print(df.to_yaml("sample-df.yaml"))

You can also load a Data Flow application directly from the YAML file saved in the
previous example:

.. code-block:: python

    df2 = Job.from_yaml(uri="sample-df.yaml")

Creating a new job and a run:

.. code-block:: python

    df_run2 = df2.create().run()

Deleting a job cancels associated runs:

.. code-block:: python

    df2.delete()
    df_run2.status

You can also load a Data Flow application from an OCID:

    df3 = Job.from_dataflow_job(df.id)

Creating a run under the same application:

.. code-block:: python

    df_run3 = df3.run()

Now there are 2 runs under the ``df`` application:

.. code-block:: python

    assert len(df.run_list()) == 2

When you run a Data Flow application, a ``DataFlowRun`` object is created.
You can check the status, wait for a run to finish, check its logs
afterwards, or cancel a run in progress. For example:

.. code-block:: python

    df_run.status
    df_run.wait()

``watch`` is an alias of ``wait``, so you can also call ``df_run.watch()``.

There are three types of logs for a run:

- application log
- driver log
- executor log

Each log consists of ``stdout`` and ``stderr``. For example, to access ``stdout``
from application log, you could use:

    df_run.logs.application.stdout

Then you could check it with:

::

   df_run.logs.application.stderr
   df_run.logs.executor.stdout
   df_run.logs.executor.stderr

You can also examine ``head`` or ``tail`` of the log, or download it to a local path. For example,

.. code-block:: python

    log = df_run.logs.application.stdout
    log.head(n=1)
    log.tail(n=1)
    log.download(<local-path>)

For the sample script, the log prints first five rows of a sample dataframe in JSON
and it looks like:

.. code-block:: python

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

.. code-block:: python

    'record 0'

Calling ``log.tail(n=1)`` returns this:

.. code-block:: python

    {"city":"Berlin","zipcode":"10437","lat_long":"52.5431572633131,13.415091104515707"}


A link to run the page in the OCI Console is given using the ``run_details_link``
property:

.. code-block:: python

    df_run.run_details_link

To list Data Flow applications, a compartment id must be given
with any optional filtering criteria. For example, you can filter by
name of the application:

.. code-block:: python

    Job.dataflow_job(compartment_id=compartment_id, display_name=name)

YAML
++++

You can create a Data Flow job directly from a YAML string. You can pass a YAML string
into the ``Job.from_yaml()`` function to build a Data Flow job:

.. code:: yaml

  kind: job
  spec:
    id: <dataflow_app_ocid>
    infrastructure:
      kind: infrastructure
      spec:
        compartmentId: <compartment_id>
        driverShape: VM.Standard.E4.Flex
        driverShapeConfig:
          ocpus: 2
          memory_in_gbs: 32
        executorShape: VM.Standard.E4.Flex
        executorShapeConfig:
          ocpus: 4
          memory_in_gbs: 64
        id: <dataflow_app_ocid>
        language: PYTHON
        logsBucketUri: <logs_bucket_uri>
        numExecutors: 1
        sparkVersion: 2.4.4
      type: dataFlow
    name: dataflow_app_name
    runtime:
      kind: runtime
      spec:
        scriptBucket: bucket_name
        scriptPathURI: oci://<bucket_name>@<namespace>/<prefix>
      type: dataFlow

**Data Flow Infrastructure YAML Schema**

.. code:: yaml

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
            driverShapeConfig:
                required: false
                type: dict
                schema:
                    ocpus:
                        required: true
                        type: float
                    memory_in_gbs:
                        required: true
                        type: float
            executorShape:
                required: false
                type: string
            executorShapeConfig:
                required: false
                type: dict
                schema:
                    ocpus:
                        required: true
                        type: float
                    memory_in_gbs:
                        required: true
                        type: float
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

.. code:: yaml

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
                    slug:
                        required: true
                        type: string
                    type:
                        allowed:
                            - service
                        required: true
                        type: string
            env:
                type: list
                required: false
                schema:
                    type: dict
            freeform_tag:
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
