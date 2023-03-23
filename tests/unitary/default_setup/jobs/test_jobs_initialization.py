#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import unittest

from ads.jobs import (
    DataScienceJob,
    DataFlow,
    ScriptRuntime,
    NotebookRuntime,
    GitPythonRuntime,
    PythonRuntime,
    ContainerRuntime,
    DataFlowRuntime,
)


class JobInitializationTest(unittest.TestCase):
    def assert_initialization(
        self, obj_class, obj_from_builder, yaml_spec, camel_spec, snake_spec
    ):
        """Asserts the object initialization
        by comparing the object initialized from:
            * YAML
            * snake case dictionary
            * snake case keyword arguments
            * camel case dictionary
            * camel case keyword arguments
        """
        self.maxDiff = None

        obj_from_yaml = obj_class.from_yaml(yaml_spec)
        self.assertEqual(obj_from_yaml._spec, obj_from_builder._spec)

        if snake_spec:
            obj_from_snake_dict = obj_class(copy.deepcopy(snake_spec))
            self.assertEqual(obj_from_yaml._spec, obj_from_snake_dict._spec)
            obj_from_snake_kwargs = obj_class(**snake_spec)
            self.assertEqual(obj_from_yaml._spec, obj_from_snake_kwargs._spec)

        if camel_spec:
            obj_from_camel_dict = obj_class(copy.deepcopy(camel_spec))
            self.assertEqual(obj_from_yaml._spec, obj_from_camel_dict._spec)
            obj_from_camel_kwargs = obj_class(**camel_spec)
            self.assertEqual(obj_from_yaml._spec, obj_from_camel_kwargs._spec)

    def test_init_data_science_job(self):
        """Test initializing DataScienceJob infrastructure."""

        yaml_spec = """
        kind: infrastructure
        type: dataScienceJob
        spec:
          logGroupId: "ocid1.loggroup.oc1..<unique_ocid>"
          logId: "ocid1.log.oc1..<unique_ocid>"
          compartmentId: "ocid1.compartment.oc1..<unique_ocid>"
          projectId: "ocid1.datascienceproject.oc1..<unique_ocid>"
          subnetId: "ocid1.subnet.oc1..<unique_ocid>"
          shapeName: "VM.Standard.E3.Flex"
          shapeConfigDetails:
            memoryInGBs: 20
            ocpus: 2
          blockStorageSize: 50
        """

        snake_spec = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "project_id": "ocid1.datascienceproject.oc1..<unique_ocid>",
            "subnet_id": "ocid1.subnet.oc1..<unique_ocid>",
            "log_group_id": "ocid1.loggroup.oc1..<unique_ocid>",
            "log_id": "ocid1.log.oc1..<unique_ocid>",
            "shape_name": "VM.Standard.E3.Flex",
            "shape_config_details": {"memory_in_gbs": 20, "ocpus": 2},
            "block_storage_size": 50,
        }

        camel_spec = {
            "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
            "projectId": "ocid1.datascienceproject.oc1..<unique_ocid>",
            "subnetId": "ocid1.subnet.oc1..<unique_ocid>",
            "logGroupId": "ocid1.loggroup.oc1..<unique_ocid>",
            "logId": "ocid1.log.oc1..<unique_ocid>",
            "shapeName": "VM.Standard.E3.Flex",
            "shapeConfigDetails": {"memoryInGBs": 20, "ocpus": 2},
            "blockStorageSize": 50,
        }

        obj_from_builder = (
            DataScienceJob()
            .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
            .with_project_id("ocid1.datascienceproject.oc1..<unique_ocid>")
            .with_subnet_id("ocid1.subnet.oc1..<unique_ocid>")
            .with_shape_name("VM.Standard.E3.Flex")
            .with_shape_config_details(memory_in_gbs=20, ocpus=2)
            .with_block_storage_size(50)
            .with_log_group_id("ocid1.loggroup.oc1..<unique_ocid>")
            .with_log_id("ocid1.log.oc1..<unique_ocid>")
        )

        self.assert_initialization(
            DataScienceJob, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_script_runtime(self):
        yaml_spec = """
        kind: runtime
        type: script
        spec:
          conda:
            slug: tensorflow26_p37_cpu_v2
            type: service
          scriptPathURI: oci://bucket_name@namespace/path/to/script.py
          args:
            - argument
            - --key
            - value
          env:
            - name: ENV
              value: value
          freeformTags:
            tag_name: tag_value
        """

        snake_spec = {
            "script_path_uri": "oci://bucket_name@namespace/path/to/script.py",
            "args": "argument --key value".split(),
            "env": {"ENV": "value"},
            "conda": {"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
            "freeform_tags": {"tag_name": "tag_value"},
        }

        camel_spec = {
            "scriptPathURI": "oci://bucket_name@namespace/path/to/script.py",
            "args": "argument --key value".split(),
            "env": {"ENV": "value"},
            "conda": {"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
            "freeformTags": {"tag_name": "tag_value"},
        }

        obj_from_builder = (
            ScriptRuntime()
            .with_source("oci://bucket_name@namespace/path/to/script.py")
            .with_service_conda("tensorflow26_p37_cpu_v2")
            .with_environment_variable(ENV="value")
            .with_argument("argument", key="value")
            .with_freeform_tag(tag_name="tag_value")
        )

        self.assert_initialization(
            ScriptRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_notebook_runtime(self):
        yaml_spec = """
        kind: runtime
        type: notebook
        spec:
          notebookPathURI: https://www.example.com/notebook.ipynb
          notebookEncoding: utf-8
          outputUri: oci://<bucket_name>@<namespace>/<prefix>
          excludeTags:
            - ignore
            - remove
          conda:
            slug: tensorflow26_p37_cpu_v2
            type: service
          env:
            - name: ENV
              value: value
        """

        snake_spec = {
            "notebook_path_uri": "https://www.example.com/notebook.ipynb",
            "output_uri": "oci://<bucket_name>@<namespace>/<prefix>",
            "notebook_encoding": "utf-8",
            "env": {"ENV": "value"},
            "conda": {"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
            "exclude_tags": ["ignore", "remove"],
        }

        camel_spec = {
            "notebookPathURI": "https://www.example.com/notebook.ipynb",
            "outputUri": "oci://<bucket_name>@<namespace>/<prefix>",
            "notebookEncoding": "utf-8",
            "env": {"ENV": "value"},
            "conda": {"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
            "excludeTags": ["ignore", "remove"],
        }

        obj_from_builder = (
            NotebookRuntime()
            .with_notebook(
                path="https://www.example.com/notebook.ipynb", encoding="utf-8"
            )
            .with_exclude_tag(["ignore", "remove"])
            .with_service_conda("tensorflow26_p37_cpu_v2")
            .with_environment_variable(ENV="value")
            .with_output("oci://<bucket_name>@<namespace>/<prefix>")
        )

        self.assert_initialization(
            NotebookRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_git_runtime(self):
        yaml_spec = """
        kind: runtime
        type: gitPython
        spec:
          entrypoint: beginner_source/examples_nn/polynomial_nn.py
          outputDir: ~/Code/tutorials/beginner_source/examples_nn
          outputUri: oci://<bucket_name>@<namespace>/<prefix>
          url: https://github.com/pytorch/tutorials.git
          conda:
            slug: pytorch19_p37_gpu_v1
            type: service
          env:
            - name: GREETINGS
              value: Welcome to OCI Data Science
        """

        snake_spec = {
            "url": "https://github.com/pytorch/tutorials.git",
            "entrypoint": "beginner_source/examples_nn/polynomial_nn.py",
            "output_uri": "oci://<bucket_name>@<namespace>/<prefix>",
            "output_dir": "~/Code/tutorials/beginner_source/examples_nn",
            "env": {"GREETINGS": "Welcome to OCI Data Science"},
            "conda": {"type": "service", "slug": "pytorch19_p37_gpu_v1"},
        }

        camel_spec = {
            "url": "https://github.com/pytorch/tutorials.git",
            "entrypoint": "beginner_source/examples_nn/polynomial_nn.py",
            "outputUri": "oci://<bucket_name>@<namespace>/<prefix>",
            "outputDir": "~/Code/tutorials/beginner_source/examples_nn",
            "env": {"GREETINGS": "Welcome to OCI Data Science"},
            "conda": {"type": "service", "slug": "pytorch19_p37_gpu_v1"},
        }

        obj_from_builder = (
            GitPythonRuntime()
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_service_conda("pytorch19_p37_gpu_v1")
            .with_source("https://github.com/pytorch/tutorials.git")
            .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
            .with_output(
                output_dir="~/Code/tutorials/beginner_source/examples_nn",
                output_uri="oci://<bucket_name>@<namespace>/<prefix>",
            )
        )

        self.assert_initialization(
            GitPythonRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_python_runtime(self):
        yaml_spec = """
        kind: runtime
        type: python
        spec:
          conda:
            slug: pytorch19_p37_cpu_v1
            type: service
          entrypoint: zip_or_dir/my_package/entry.py
          scriptPathURI: path/to/zip_or_dir
          workingDir: zip_or_dir
          outputDir: zip_or_dir/output
          outputUri: oci://<bucket_name>@<namespace>/<prefix>
          pythonPath:
            - "my_python_packages"
        """

        snake_spec = {
            "script_path_uri": "path/to/zip_or_dir",
            "entrypoint": "zip_or_dir/my_package/entry.py",
            "output_uri": "oci://<bucket_name>@<namespace>/<prefix>",
            "output_dir": "zip_or_dir/output",
            "working_dir": "zip_or_dir",
            "python_path": ["my_python_packages"],
            "conda": {"type": "service", "slug": "pytorch19_p37_cpu_v1"},
        }

        camel_spec = {
            "scriptPathURI": "path/to/zip_or_dir",
            "entrypoint": "zip_or_dir/my_package/entry.py",
            "outputUri": "oci://<bucket_name>@<namespace>/<prefix>",
            "outputDir": "zip_or_dir/output",
            "workingDir": "zip_or_dir",
            "pythonPath": ["my_python_packages"],
            "conda": {"type": "service", "slug": "pytorch19_p37_cpu_v1"},
        }

        obj_from_builder = (
            PythonRuntime()
            .with_service_conda("pytorch19_p37_cpu_v1")
            .with_source(
                "path/to/zip_or_dir", entrypoint="zip_or_dir/my_package/entry.py"
            )
            .with_working_dir("zip_or_dir")
            .with_python_path("my_python_packages")
            .with_output(
                "zip_or_dir/output", "oci://<bucket_name>@<namespace>/<prefix>"
            )
        )

        self.assert_initialization(
            PythonRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_container_runtime(self):
        yaml_spec = """
        kind: runtime
        type: container
        spec:
          image: <region>.ocir.io/<your_tenancy>/<your_image>
          cmd:
            - sleep 5 && echo $GREETINGS
          entrypoint:
            - /bin/sh
            - -c
          env:
            - name: GREETINGS
              value: Welcome to OCI Data Science
        """

        snake_spec = {
            "image": "<region>.ocir.io/<your_tenancy>/<your_image>",
            "entrypoint": ["/bin/sh", "-c"],
            "cmd": ["sleep 5 && echo $GREETINGS"],
            "env": {"GREETINGS": "Welcome to OCI Data Science"},
        }

        camel_spec = {
            "image": "<region>.ocir.io/<your_tenancy>/<your_image>",
            "entrypoint": ["/bin/sh", "-c"],
            "cmd": ["sleep 5 && echo $GREETINGS"],
            "env": {"GREETINGS": "Welcome to OCI Data Science"},
        }

        obj_from_builder = (
            ContainerRuntime()
            .with_image("<region>.ocir.io/<your_tenancy>/<your_image>")
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_entrypoint(["/bin/sh", "-c"])
            .with_cmd(["sleep 5 && echo $GREETINGS"])
        )

        self.assert_initialization(
            ContainerRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_dataflow_runtime(self):
        yaml_spec = """
        kind: runtime
        type: dataFlow
        spec:
          scriptBucket: bucket_name
          scriptPathURI: oci://<bucket_name>@<namespace>/<prefix>
          overwrite: True
        """

        snake_spec = {
            "script_bucket": "bucket_name",
            "script_path_uri": "oci://<bucket_name>@<namespace>/<prefix>",
            "overwrite": True,
        }

        camel_spec = {
            "scriptBucket": "bucket_name",
            "scriptPathURI": "oci://<bucket_name>@<namespace>/<prefix>",
            "overwrite": True,
        }

        obj_from_builder = (
            DataFlowRuntime()
            .with_script_uri("oci://<bucket_name>@<namespace>/<prefix>")
            .with_script_bucket("bucket_name")
            .with_overwrite(True)
        )

        self.assert_initialization(
            DataFlowRuntime, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )

    def test_init_dataflow(self):
        yaml_spec = """
        kind: runtime
        type: dataFlow
        spec:
          compartmentId: ocid1.compartment.oc1..<unique_ocid>
          driverShape: VM.Standard2.1
          executorShape: VM.Standard2.1
          logsBucketUri: oci://<bucket_name>@<namespace>/<prefix>
          numExecutors: 1
          sparkVersion: 3.2.1
        """

        snake_spec = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "driver_shape": "VM.Standard2.1",
            "executor_shape": "VM.Standard2.1",
            "logs_bucket_uri": "oci://<bucket_name>@<namespace>/<prefix>",
            "num_executors": 1,
            "spark_version": "3.2.1",
        }

        camel_spec = {
            "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
            "driverShape": "VM.Standard2.1",
            "executorShape": "VM.Standard2.1",
            "logsBucketUri": "oci://<bucket_name>@<namespace>/<prefix>",
            "numExecutors": 1,
            "sparkVersion": "3.2.1",
        }

        obj_from_builder = (
            DataFlow()
            .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
            .with_logs_bucket_uri("oci://<bucket_name>@<namespace>/<prefix>")
            .with_driver_shape("VM.Standard2.1")
            .with_executor_shape("VM.Standard2.1")
            .with_spark_version("3.2.1")
        )

        self.assert_initialization(
            DataFlow, obj_from_builder, yaml_spec, camel_spec, snake_spec
        )
