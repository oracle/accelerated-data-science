===========
Quick Start
===========

Data Flow is a hosted Apache Spark server. It is quick to start, and can scale to handle large datasets in parallel. ADS provides a convenient API for creating and maintaining workloads on Data Flow.

Submit a Dummy Python Script to DataFlow
========================================

From a Python Environment
-------------------------

Submit a python script to DataFlow entirely from your python environment. 
The following snippet uses a dummy python script that prints "Hello World" 
followed by the spark version, 3.2.1.

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
			.with_logs_bucket_uri("oci://<mybucket>@<mynamespace>/<dataflow-logs-prefix>")
			.with_driver_shape("VM.Standard.E4.Flex")
			.with_driver_shape_config(ocpus=2, memory_in_gbs=32)
			.with_executor_shape("VM.Standard.E4.Flex")
			.with_executor_shape_config(ocpus=4, memory_in_gbs=64)
			.with_spark_version("3.2.1")
		)
		runtime_config = (
			DataFlowRuntime()
			.with_script_uri(os.path.join(td, "script.py"))
			.with_script_bucket("oci://<mybucket>@<mynamespace>/<subdir_to_put_and_get_script>")
		)
		df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
		df.create()
		df_run = df.run()

From the Command Line
---------------------

The same result can be achieved from the command line using ``ads CLI`` and a yaml file.

Assuming you have the following two files written in your current directory as ``script.py`` and ``dataflow.yaml`` respectively:


.. code-block:: python

	# script.py
	import pyspark
	def main():
		print("Hello World")
		print("Spark version is", pyspark.__version__)
	if __name__ == "__main__":
		main()


.. code-block:: yaml

	# dataflow.yaml
	kind: job
	spec:
		name: dataflow-app-<uuid>
		infrastructure:
			kind: infrastructure
			spec:
				compartmentId: oci.xx.<compartment_id>
				logsBucketUri: oci://<mybucket>@<mynamespace>/<dataflow-logs-prefix>
				driverShape: VM.Standard.E4.Flex
                driverShapeConfig:
                  ocpus: 2
                  memory_in_gbs: 32
                executorShape: VM.Standard.E4.Flex
                executorShapeConfig:
                  ocpus: 4
                  memory_in_gbs: 64
				sparkVersion: 3.2.1
				numExecutors: 1
			type: dataFlow
		runtime:
			kind: runtime
			spec:
				scriptUri: script.py
				scriptBucket: oci://<mybucket>@<mynamespace>/<subdir_to_put_and_get_script>


.. code-block:: shell

	ads jobs run -f dataflow.yaml


Real Data Flow Example with Conda Environment
=============================================

From PySpark v3.0.0 and onwards, Data Flow allows a published conda environment as the `Spark runtime environment <https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-conda>`_ when built with `ADS`. Data Flow supports published conda environments only. Conda packs are tar'd conda environments. When you publish your own conda packs to object storage, ensure that the DataFlow Resource has access to read the object or bucket.
Below is a more built-out example using conda packs:

From a Python Environment
-------------------------

.. code-block:: python

	from ads.jobs import DataFlow, Job, DataFlowRuntime
	from uuid import uuid4
	import os
	import tempfile

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
		Create a Spark session
		spark = SparkSession.builder.appName(app_name).getOrCreate()

		Load a csv file from dataflow public storage
		df = (
			spark.read.format("csv")
			.option("header", "true")
			.option("multiLine", "true")
			.load(
				"oci://oow_2019_dataflow_lab@bigdatadatasciencelarge/usercontent/kaggle_berlin_airbnb_listings_summary.csv"
			)
		)

		Create a temp view and do some SQL operations
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
			.with_compartment_id("oci.xx.<compartment_id>")
			.with_logs_bucket_uri("oci://<mybucket>@<mynamespace>/<dataflow-logs-prefix>")
			.with_driver_shape("VM.Standard.E4.Flex")
			.with_driver_shape_config(ocpus=2, memory_in_gbs=32)
			.with_executor_shape("VM.Standard.E4.Flex")
			.with_executor_shape_config(ocpus=4, memory_in_gbs=64)
			.with_spark_version("3.2.1")
		)
		runtime_config = (
			DataFlowRuntime()
			.with_script_uri(os.path.join(td, "script.py"))
			.with_script_bucket("oci://<mybucket>@<mynamespace>/<subdir_to_put_and_get_script>")
			.with_custom_conda(uri="oci://<mybucket>@<mynamespace>/<path_to_conda_pack>")
			.with_arguments(["run-test", "-v", "-l", "5"])
		)
		df = Job(name=name, infrastructure=dataflow_configs, runtime=runtime_config)
		df.create()
		df_run = df.run()


From the Command Line
---------------------

Again, assume you have the following two files written in your current directory as ``script.py`` and ``dataflow.yaml`` respectively:

.. code-block:: python
   
	# script.py
	from pyspark.sql import SparkSession
	import click

	@click.command()
	@click.argument("app_name")
	@click.option(
		"--limit", "-l", help="max number of row to print", default=10, required=False
	)
	@click.option("--verbose", "-v", help="print out result in verbose mode", is_flag=True)
	def main(app_name, limit, verbose):
		Create a Spark session
		spark = SparkSession.builder.appName(app_name).getOrCreate()

		Load a csv file from dataflow public storage
		df = (
			spark.read.format("csv")
			.option("header", "true")
			.option("multiLine", "true")
			.load(
				"oci://oow_2019_dataflow_lab@bigdatadatasciencelarge/usercontent/kaggle_berlin_airbnb_listings_summary.csv"
			)
		)

		Create a temp view and do some SQL operations
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


.. code-block:: yaml
   
	# dataflow.yaml
	kind: job
	spec:
		name: dataflow-app-<uuid>
		infrastructure:
			kind: infrastructure
			spec:
				compartmentId: oci.xx.<compartment_id>
				logsBucketUri: oci://<mybucket>@<mynamespace>/<dataflow-logs-prefix>
				driverShape: VM.Standard.E4.Flex
				driverShapeConfig:
					ocpus: 2
					memory_in_gbs: 32
				executorShape: VM.Standard.E4.Flex
				executorShapeConfig:
					ocpus: 4
					memory_in_gbs: 64
				sparkVersion: 3.2.1
				numExecutors: 1
			type: dataFlow
		runtime:
			kind: runtime
			spec:
				scriptUri: script.py
				scriptBucket: oci://<mybucket>@<mynamespace>/<subdir_to_put_and_get_script>
				conda:
					uri: oci://<mybucket>@<mynamespace>/<path_to_conda_pack>
					type: published
				args:
					- "run-test"
					- "-v"
					- "-l"
					- "5"


.. code-block:: shell

	ads jobs run -f dataflow.yaml