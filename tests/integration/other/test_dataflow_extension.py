#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import builtins

import pytest
from IPython.testing.globalipapp import start_ipython
from IPython.utils.io import capture_output

from ads.jobs.utils import get_dataflow_config, DataFlowConfig
from ads.jobs.extension import dataflow
from tests.integration.config import secrets


class TestDataFlowExt:
    @pytest.fixture(scope="class")
    def ip(self):
        start_ipython()
        ip = builtins.ip
        ip.run_line_magic("load_ext", "ads.jobs.extension")
        yield ip

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "dataflow_config.ini"), "w") as f:
                f.write(
                    """
[PROFILE_NAME]
property1: value1
property2: value2
                """
                )
            dataflow_config = get_dataflow_config(
                path=os.path.join(td, "dataflow_config.ini"), oci_profile="PROFILE_NAME"
            )
            assert dataflow_config["property1"] == "value1"
            assert dataflow_config["property2"] == "value2"
            with pytest.raises(ValueError):
                get_dataflow_config(
                    path=os.path.join(td, "dataflow_config.ini"),
                    oci_profile="NON_EXIST",
                )
            assert len(get_dataflow_config(path=os.path.join(td, "non_exist.ini"))) == 0

    @pytest.mark.skip("ODSC-31213: failing")
    def test_create_and_run(self, ip):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        dataflow_config = DataFlowConfig(
            path=os.path.join(curr_dir, "opctl_tests_files", "dataflow_config.ini")
        )
        dataflow_config.num_executors = 2
        dataflow_config.configuration = {"spark.driver.memory": "512m"}

        cell = """
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
    query_result_df = spark.sql("SELECT city, zipcode, CONCAT(latitude,',', longitude) AS lat_long FROM berlin").limit(limit)

    # Convert the filtered Spark DataFrame into JSON format
    # Note: we are writing to the spark stdout log so that we can retrieve the log later at the end of the notebook.
    if verbose:
        rows = query_result_df.toJSON().collect()
        for i, row in enumerate(rows):
            print(f"record {i}")
            print(row)


if __name__ == "__main__":
    main()
        """
        # ip.run_line_magic("load_ext", "ads.jobs.extension")
        with capture_output() as captured:
            line = f"run -o -f my_test_script.py -w -a oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip -c {dataflow_config} -- abc -l 5 -v"
            ip.run_cell_magic("dataflow", line, cell)
        stdout = captured.stdout
        assert "SUCCEEDED" in stdout, stdout

        # remove negative test to avoid prod alert
        # with capture_output() as captured:
        #     line = f"run -o -f my_test_script.py -w -a oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip -c {dataflow_config}"
        #     ip.run_cell_magic("dataflow", line, cell)
        # stdout = captured.stdout
        # assert "FAILED" in stdout, stdout

        with pytest.raises(FileExistsError):
            line = f"run -f my_test_script.py -w -a oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip -c {dataflow_config} -- abc -l 5 -v"
            ip.run_cell_magic("dataflow", line, cell)

    def test_dataflow_log(self, ip):
        with capture_output() as captured:
            line = f"log -n 3 {secrets.other.DATAFLOW_RUN_ID}"
            ip.run_line_magic("dataflow", line)
        stdout = captured.stdout
        assert len(stdout.strip("\n").split("\n")) == 3, stdout

        with pytest.raises(ValueError):
            line = "log"
            ip.run_line_magic("dataflow", line)

    def test_dataflow_help(self, ip):
        with capture_output() as captured:
            ip.run_line_magic("dataflow", "-h")
        stdout = captured.stdout
        assert (
            "Run `dataflow run -h` or `dataflow log -h` to see options for subcommands."
            in stdout
        )
        with capture_output() as captured:
            ip.run_line_magic("dataflow", "run -h")
        stdout = captured.stdout
        assert "Usage: dataflow run [OPTIONS] -- [ARGS]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("dataflow", "log -h")
        stdout = captured.stdout
        assert "Usage: dataflow log [OPTIONS] [RUN_ID]" in stdout
