
Data Flow
*********

This example demonstrates how to create a Data Flow application that is connected to the Data Catalog Metastore. It creates a PySpark script, then a Data Flow application. This application can be run by directly by Data Flow or as part of a Job.

This section runs Hive queries using Data Flow. When the Data Catalog is being used the only changes that need to be made are to provide the metastore OCID.

PySpark Script
==============

A PySpark script is needed for the Data Flow application. The following code creates that script. The script will use Spark to load a CSV file from a public Object Storage bucket. It will then create a database and write the file to Object Storage. Finally, it will use Spark SQL to query the database and print the records in JSON format. 

There is nothing in the PySpark script that is specific to using Data Catalog Metastore. The script treats the database as a standard Hive database.

.. code-block:: python3

    script = '''
    from pyspark.sql import SparkSession

    def main():   
        
        database_name = "employee_attrition"
        table_name = "orcl_attrition"
        
        # Create a Spark session
        spark = SparkSession \\
            .builder \\
            .appName("Python Spark SQL basic example") \\
            .enableHiveSupport() \\
            .getOrCreate()
        
        # Load a CSV file from a public Object Storage bucket
        df = spark \\
            .read \\
            .format("csv") \\
            .option("header", "true") \\
            .option("multiLine", "true") \\
            .load("oci://hosted-ds-datasets@bigdatadatasciencelarge/synthetic/orcl_attrition.csv")
            
        print(f"Creating {database_name}")
        spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

        # Write the data to the database
        df.write.mode("overwrite").saveAsTable(f"{database_name}.{table_name}")
        
        # Use Spark SQL to read from the database.
        query_result_df = spark.sql(f"""
                                    SELECT EducationField, SalaryLevel, JobRole FROM {database_name}.{table_name} limit 10
                                    """)

        # Convert the filtered Apache Spark DataFrame into JSON format and write it out to stdout
        # so that it can be captured in the log.
        print('\\n'.join(query_result_df.toJSON().collect()))

    if __name__ == '__main__':
        main()
    '''

    # Save the PySpark script to a file
    dataflow_base_folder = tempfile.mkdtemp()
    script_uri = os.path.join(dataflow_base_folder, "example.py")
    
    with open(script_uri, 'w') as f:
        print(script.strip(), file=f)

Create Application
==================


To create a Data Flow application you will need ``DataFlow`` and ``DataFlowRuntime`` objects. A ``DataFlow`` object stores the properties that are specific to the Data Flow service. These would be things such as the compartment OCID, the URI to the Object Storage bucket for the logs, the type of hardware to be used, the version of Spark, and much more. If you are using a Data Catalog Metastore to manage a database, the metastore OCID is stored in this object. The ``DataFlowRuntime`` object stores properties related to the script to be run. This would be the bucket to be used for the script, the location of the PySpark script, and any command-line arguments.

Update the ``script_bucket``, ``log_bucket``, and ``metastore_id``  variables to match your tenancy’s configuration.

.. code-block:: python3

    # Update values
    log_bucket_uri = "oci://<bucket_name>@<namespace>/<prefix>"
    metastore_id = "<metastore_id>"
    script_bucket = "oci://<bucket_name>@<namespace>/<prefix>"

    compartment_id = os.environ.get("NB_SESSION_COMPARTMENT_OCID")
    driver_shape = "VM.Standard.E4.Flex"
    driver_shape_config = {"ocpus":2, "memory_in_gbs":32}
    executor_shape = "VM.Standard.E4.Flex"
    executor_shape_config = {"ocpus":4, "memory_in_gbs":64}
    spark_version = "3.2.1"

In the following example, a ``DataFlow`` is created and populated with the information that it needs to define the Data Flow service. Since, we are connecting to the Data Catalog Metastore to work with a Hive database, the metastore OCID must be given.

.. code-block:: python3

    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime

    dataflow_configs = DataFlow(
        {"compartment_id": compartment_id,
         "driver_shape": driver_shape,
         "driver_shape_config": driver_shape_config,
         "executor_shape": executor_shape,
         "executor_shape_config": executor_shape_config,
         "logs_bucket_uri": log_bucket_uri,
         "metastore_id": metastore_id,
         "spark_version": spark_version}
    )
 
In the following example, a ``DataFlowRuntime`` is created and populated with the URI to the PySpark script and the URI for the script bucket. The script URI specifies the path to the script. It can be local or remote (an Object Storage path). If the path is local, then a URI to the script bucket must also be specified. This is because Data Flow requires a script to be in Object Storage. If the specified path to the PySpark script is on a local drive, ADS will upload it for you.

.. code-block:: python3

    runtime_config = DataFlowRuntime(
        {
            "script_bucket": script_uri
            "script_uri": pyspark_file_path,
        }
    )

Run
===

The recommended approach for running Data Flow applications is to use a Job. This will prevent your notebook from being blocked. 

A Job requires a name, infrastructure, and runtime settings. Update the following code to give the job a unique name. The ``infrastructure`` takes a ``DataFlow`` object and the ``runtime`` parameter takes a ``DataFlowRuntime`` object.

.. code-block:: python3

    # Update values
    job_name = "<job_name>"

    df_job = Job(name=job_name, 
                 infrastructure=dataflow_configs, 
                 runtime=runtime_config)
    df_app = df_job.create()
    df_run = df_app.run()

