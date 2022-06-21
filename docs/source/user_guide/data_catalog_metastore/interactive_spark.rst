Interactive Spark 
*****************

This section demonstrates how to make connections to the Data Catalog Metastore and Object Storage. It uses Spark to load data from a public Object Storage file and creates a database. The metadata for the database is managed by the Data Catalog Metastore and the data is copied to your data warehouse bucket. Finally, Spark is used to make a Spark SQL query on the database.

Specify the bucket URI that will act as the data warehouse. Use the ``warehouse_uri`` variable and it should have the following format ``oci://<bucket_name>@<namespace_name>/<prefix>``.  Update the variable ``metastore_id`` with the OCID of the Data Catalog Metastore. 

Create a Spark session that connects to the Data Catalog Metastore and the Object Storage that will act as the data warehouse.

.. code-block:: python3

    from pyspark.sql import SparkSession

    warehouse_uri = "<warehouse_uri>"
    metastore_id = "<metastore_id>"

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Hive integration example") \
        .config("spark.sql.warehouse.dir", warehouse_uri) \
        .config("spark.hadoop.oracle.dcat.metastore.id", metastore_id) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
Load a data file from Object Storage into a Spark DataFrame. Create a database in the Data Catalog Metastore and then save the dataframe as a table. This will write the files to the location specified by the ``warehouse_uri`` variable. 

.. code-block:: python3

    database_name = "ODSC_DEMO"
    table_name = "ODSC_PYSPARK_METASTORE_DEMO"
    file_path = "oci://hosted-ds-datasets@bigdatadatasciencelarge/synthetic/orcl_attrition.csv"

    input_dataframe = spark.read.option("header", "true").csv(file_path)
    spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
    spark.sql(f"CREATE DATABASE {database_name}")
    input_dataframe.write.mode("overwrite").saveAsTable(f"{database_name}.{table_name}")

Use Spark SQL to read from the database.

.. code-block:: python3

    spark_df = spark.sql(f"""
                         SELECT EducationField, SalaryLevel, JobRole FROM {database_name}.{table_name} limit 10
                         """) 
    spark_df.show()

