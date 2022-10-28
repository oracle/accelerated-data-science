Quick Start
***********

Data Flow
=========

.. code-block:: python

    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime

    # Update these values
    job_name = "<job_name>"
    logs_bucket = "oci://<bucket_name>@<namespace>/<prefix>"
    metastore_id = "<metastore_id>"
    script_bucket = "oci://<bucket_name>@<namespace>/<prefix>"

    compartment_id = os.environ.get("NB_SESSION_COMPARTMENT_OCID")
    driver_shape = "VM.Standard.E4.Flex"
    driver_shape_config = {"ocpus":2, "memory_in_gbs":32}
    executor_shape = "VM.Standard.E4.Flex"
    executor_shape_config = {"ocpus":4, "memory_in_gbs":64}
    spark_version = "3.2.1"

    # A python script to be run in Data Flow
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

    # Saves the python script to local path.
    dataflow_base_folder = tempfile.mkdtemp()
    script_uri = os.path.join(dataflow_base_folder, "example.py")
    
    with open(script_uri, 'w') as f:
        print(script.strip(), file=f)

    dataflow_configs = DataFlow(
        {
            "compartment_id": compartment_id,
            "driver_shape": driver_shape,
            "driver_shape_config": driver_shape_config,
            "executor_shape": executor_shape,
            "executor_shape_config": executor_shape_config,
            "logs_bucket_uri": log_bucket_uri,
            "metastore_id": metastore_id,
            "spark_version": spark_version
        }
    )

    runtime_config = DataFlowRuntime(
        {
            "script_uri": pyspark_file_path,
            "script_bucket": script_uri
        }
    )
    
    # creates a Data Flow application with DataFlow and DataFlowRuntime.
    df_job = Job(name=job_name, 
                 infrastructure=dataflow_configs, 
                 runtime=runtime_config)
    df_app = df_job.create()
    df_run = df_app.run()

    # check a job log
    df_run.watch()


Interactive Spark 
=================

.. code-block:: python3

    from pyspark.sql import SparkSession

    # Update these values
    warehouse_uri = "<warehouse_uri>"
    metastore_id = "<metastore_id>"

    database_name = "ODSC_DEMO"
    table_name = "ODSC_PYSPARK_METASTORE_DEMO"

    # create a spark session
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Hive integration example") \
        .config("spark.sql.warehouse.dir", warehouse_uri) \
        .config("spark.hadoop.oracle.dcat.metastore.id", metastore_id) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # show the databases in the warehouse:
    spark.sql("SHOW DATABASES").show()
    spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
    spark.sql(f"CREATE DATABASE {database_name}")

    # Load the Employee Attrition data file from OCI Object Storage into a Spark DataFrame:
    file_path = "oci://hosted-ds-datasets@bigdatadatasciencelarge/synthetic/orcl_attrition.csv"
    input_dataframe = spark.read.option("header", "true").csv(file_path)
    input_dataframe.write.mode("overwrite").saveAsTable(f"{database_name}.{table_name}")

    # explore data
    spark_df = spark.sql(f"""
                         SELECT EducationField, SalaryLevel, JobRole FROM {database_name}.{table_name} limit 10
                         """) 
    spark_df.show()


