#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
import yaml

from ads.jobs.schema.validator import (
    ValidatorFactory,
    ValidateJob,
    ValidateInfrastructure,
    ValidateRuntime,
)


class TestValidators:
    def test_validate_job(self):
        dict_file = yaml.safe_load(
            f"""
        kind: job
        spec:
            id: ocid1.datasciencejob.oc1.iad.<unique_ocid>
            infrastructure:
                kind: infrastructure
                spec:
                    blockStorageSize: 50
                    compartmentId: ocid1.compartment.oc1..<unique_ocid>
                    displayName: ads_my_script
                    jobInfrastructureType: STANDALONE
                    jobType: DEFAULT
                    projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                    shapeName: VM.Standard2.1
                    subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
                    logId: ocid1.log.oc1.iad.<unique_ocid>
                type: dataScienceJob
            name: ads_my_script
            runtime:
                kind: runtime
                spec:
                    args:
                        - pos_arg1
                        - pos_arg2
                        - --key1
                        - val1
                        - --key2
                        - val2
                    conda:
                        slug: mlcpuv1
                        type: service
                    env:
                        - name: KEY1
                          value: VALUE1
                        - name: KEY2
                          value: VALUE2
                    scriptPathURI: ads_my_script.py
                type: python"""
        )
        v = ValidateJob(dict_file).validate()
        expected_normalized_dict = {
            "kind": "job",
            "spec": {
                "id": "ocid1.datasciencejob.oc1.iad.<unique_ocid>",
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {
                        "blockStorageSize": 50,
                        "logId": "ocid1.log.oc1.iad.<unique_ocid>",
                        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                        "displayName": "ads_my_script",
                        "jobInfrastructureType": "STANDALONE",
                        "jobType": "DEFAULT",
                        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "shapeName": "VM.Standard2.1",
                        "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    },
                    "type": "dataScienceJob",
                },
                "name": "ads_my_script",
                "runtime": {
                    "kind": "runtime",
                    "spec": {
                        "args": [
                            "pos_arg1",
                            "pos_arg2",
                            "--key1",
                            "val1",
                            "--key2",
                            "val2",
                        ],
                        "conda": {"slug": "mlcpuv1", "type": "service"},
                        "env": [
                            {"name": "KEY1", "value": "VALUE1"},
                            {"name": "KEY2", "value": "VALUE2"},
                        ],
                        "scriptPathURI": "ads_my_script.py",
                    },
                    "type": "python",
                },
            },
        }
        assert v == expected_normalized_dict

    def test_validate_full_python_runtime(self):
        dict_file = yaml.safe_load(
            f"""
        apiVersion: null
        kind: job
        spec:
            infrastructure:
                kind: infrastructure
                spec:
                    blockStorageSize: 50
                    compartmentId: ocid1.compartment.oc1..<unique_ocid>
                    displayName: ads_my_script
                    jobInfrastructureType: STANDALONE
                    jobType: DEFAULT
                    projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                    shapeName: VM.Standard2.1
                    subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
                    logId: ocid1.log.oc1.iad.<unique_ocid>
                type: dataScienceJob
            name: ads_my_script
            runtime:
                kind: runtime
                spec:
                    args:
                        - pos_arg1
                        - pos_arg2
                        - --key1
                        - val1
                        - --key2
                        - val2
                    conda:
                        slug: mlcpuv1
                        type: service
                    env:
                        - name: KEY1
                          value: VALUE1
                        - name: KEY2
                          value: VALUE2
                    scriptPathURI: ads_my_script.py
                type: python
        """
        )
        v = ValidateJob(dict_file).validate()
        expected_normalized_dict = {
            "apiVersion": None,
            "kind": "job",
            "spec": {
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {
                        "blockStorageSize": 50,
                        "logId": "ocid1.log.oc1.iad.<unique_ocid>",
                        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                        "displayName": "ads_my_script",
                        "jobInfrastructureType": "STANDALONE",
                        "jobType": "DEFAULT",
                        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "shapeName": "VM.Standard2.1",
                        "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    },
                    "type": "dataScienceJob",
                },
                "name": "ads_my_script",
                "runtime": {
                    "kind": "runtime",
                    "spec": {
                        "args": [
                            "pos_arg1",
                            "pos_arg2",
                            "--key1",
                            "val1",
                            "--key2",
                            "val2",
                        ],
                        "conda": {"slug": "mlcpuv1", "type": "service"},
                        "env": [
                            {"name": "KEY1", "value": "VALUE1"},
                            {"name": "KEY2", "value": "VALUE2"},
                        ],
                        "scriptPathURI": "ads_my_script.py",
                    },
                    "type": "python",
                },
            },
        }
        assert v == expected_normalized_dict

    def test_validate_notebook_runtime(self):
        dict_file = yaml.safe_load(
            f"""
        apiVersion: null
        kind: job
        spec:
            infrastructure:
                kind: infrastructure
                spec:
                    blockStorageSize: 50
                    compartmentId: ocid1.compartment.oc1..<unique_ocid>
                    displayName: ads_my_script
                    jobInfrastructureType: STANDALONE
                    jobType: DEFAULT
                    projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                    shapeName: VM.Standard2.1
                    subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
                    logId: ocid1.log.oc1.iad.<unique_ocid>
                type: dataScienceJob
            name: ads_my_script
            runtime:
                kind: runtime
                spec:
                    args:
                        - pos_arg1
                        - pos_arg2
                        - --key1
                        - val1
                        - --key2
                        - val2
                    env:
                        - name: KEY1
                          value: VALUE1
                        - name: KEY2
                          value: VALUE2
                    notebookPathURI: oci://bucket/folder/mynotebook.ipynb
                    excludeTags:
                        - test
                        - plot
                type: notebook"""
        )
        v = ValidateJob(dict_file).validate()
        expected_normalized_dict = {
            "apiVersion": None,
            "kind": "job",
            "spec": {
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {
                        "blockStorageSize": 50,
                        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                        "logId": "ocid1.log.oc1.iad.<unique_ocid>",
                        "displayName": "ads_my_script",
                        "jobInfrastructureType": "STANDALONE",
                        "jobType": "DEFAULT",
                        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "shapeName": "VM.Standard2.1",
                        "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    },
                    "type": "dataScienceJob",
                },
                "name": "ads_my_script",
                "runtime": {
                    "kind": "runtime",
                    "spec": {
                        "args": [
                            "pos_arg1",
                            "pos_arg2",
                            "--key1",
                            "val1",
                            "--key2",
                            "val2",
                        ],
                        "env": [
                            {"name": "KEY1", "value": "VALUE1"},
                            {"name": "KEY2", "value": "VALUE2"},
                        ],
                        "notebookPathURI": "oci://bucket/folder/mynotebook.ipynb",
                        "excludeTags": ["test", "plot"],
                    },
                    "type": "notebook",
                },
            },
        }
        assert v == expected_normalized_dict

    def test_validate_git_runtime(self):
        dict_file = yaml.safe_load(
            f"""
        apiVersion: null
        kind: job
        spec:
            infrastructure:
                kind: infrastructure
                spec:
                    blockStorageSize: 50
                    compartmentId: ocid1.compartment.oc1..<unique_ocid>
                    displayName: ads_my_script
                    jobInfrastructureType: STANDALONE
                    jobType: DEFAULT
                    projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                    shapeName: VM.Standard2.1
                    subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
                type: dataScienceJob
            name: ads_my_script
            runtime:
                kind: runtime
                spec:
                    url: https://github.com/test_repo/jobs-scripts.git
                    branch: main
                    entrypoint: xgboost_classification_onnx_clean.py
                    entryFunction: main
                    params:
                        dataPath: oci://test_bucket/jobs/etl-out/*.parquet
                    codeDir: /home/datascience/app/code
                    outputDir: /home/datascience/output
                    outputUri: oci://test_bucket@test_namespace/jobs-git-boot-loader
                type: gitPython"""
        )
        v = ValidateJob(dict_file).validate()
        expected_normalized_dict = {
            "apiVersion": None,
            "kind": "job",
            "spec": {
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {
                        "blockStorageSize": 50,
                        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                        "displayName": "ads_my_script",
                        "jobInfrastructureType": "STANDALONE",
                        "jobType": "DEFAULT",
                        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "shapeName": "VM.Standard2.1",
                        "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    },
                    "type": "dataScienceJob",
                },
                "name": "ads_my_script",
                "runtime": {
                    "kind": "runtime",
                    "spec": {
                        "url": "https://github.com/test_repo/jobs-scripts.git",
                        "branch": "main",
                        "entrypoint": "xgboost_classification_onnx_clean.py",
                        "entryFunction": "main",
                        "params": {
                            "dataPath": "oci://test_bucket/jobs/etl-out/*.parquet"
                        },
                        "codeDir": "/home/datascience/app/code",
                        "outputDir": "/home/datascience/output",
                        "outputUri": "oci://test_bucket@test_namespace/jobs-git-boot-loader",
                    },
                    "type": "gitPython",
                },
            },
        }
        assert v == expected_normalized_dict

    def test_validate_container_runtime(self):
        dict_file = yaml.safe_load(
            f"""
        apiVersion: null
        kind: job
        spec:
            infrastructure:
                kind: infrastructure
                spec:
                    blockStorageSize: 50 # validate is float min 50
                    compartmentId: ocid1.compartment.oc1..<unique_ocid> # if you're feeling courageous you can add a customer cerberus validator to validate ocids against a regex pattern
                    displayName: ads_my_script
                    jobInfrastructureType: STANDALONE
                    jobType: DEFAULT # validate from list of alloweds
                    projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                    shapeName: VM.Standard2.1 # don't validate, only type as string
                    subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
                type: dataScienceJob # valiudate from allowed list of dataScienceJob|dataFlowJob|kubernetesJob (for now)
            name: ads_my_script
            runtime:
                kind: runtime # validate as being this
                spec:
                    args:
                        - print
                        - bpi(2000)
                    command:
                        - perl
                        - -Mbignum=bpi
                        - wle
                    env:
                        - name: KEY1
                          value: VALUE1
                        - name: KEY2
                          value: VALUE2
                    image: iad.ocir.io/myimage:2.0.1
                type: container"""
        )
        v = ValidateJob(dict_file).validate()
        expected_normalized_dict = {
            "apiVersion": None,
            "kind": "job",
            "spec": {
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {
                        "blockStorageSize": 50,
                        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                        "displayName": "ads_my_script",
                        "jobInfrastructureType": "STANDALONE",
                        "jobType": "DEFAULT",
                        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "shapeName": "VM.Standard2.1",
                        "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    },
                    "type": "dataScienceJob",
                },
                "name": "ads_my_script",
                "runtime": {
                    "kind": "runtime",
                    "spec": {
                        "args": ["print", "bpi(2000)"],
                        "command": ["perl", "-Mbignum=bpi", "wle"],
                        "env": [
                            {"name": "KEY1", "value": "VALUE1"},
                            {"name": "KEY2", "value": "VALUE2"},
                        ],
                        "image": "iad.ocir.io/myimage:2.0.1",
                    },
                    "type": "container",
                },
            },
        }
        assert v == expected_normalized_dict

    def test_validate_infra_ds_job(self):
        dict_file = yaml.safe_load(
            f"""
        kind: infrastructure
        spec:
            blockStorageSize: 50
            compartmentId: ocid1.compartment.oc1..<unique_ocid>
            displayName: ads_my_script
            jobInfrastructureType: STANDALONE
            jobType: DEFAULT
            projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
            shapeName: VM.Standard2.1
            subnetId: ocid1.subnet.oc1.iad.<unique_ocid>
        type: dataScienceJob
        """
        )
        v = ValidateInfrastructure(dict_file).validate()
        expected_normalized_dict = {
            "kind": "infrastructure",
            "spec": {
                "blockStorageSize": 50,
                "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
                "displayName": "ads_my_script",
                "jobInfrastructureType": "STANDALONE",
                "jobType": "DEFAULT",
                "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                "shapeName": "VM.Standard2.1",
                "subnetId": "ocid1.subnet.oc1.iad.<unique_ocid>",
            },
            "type": "dataScienceJob",
        }
        assert v == expected_normalized_dict

    def test_validate_runtime_python(self):
        dict_file = yaml.safe_load(
            f"""
        kind: runtime
        spec:
            args:
                - pos_arg1
                - pos_arg2
                - --key1
                - val1
                - --key2
                - val2
            conda:
                slug: mlcpuv1
                type: service
            env:
                - name: KEY1
                  value: VALUE1
                - name: KEY2
                  value: VALUE2
            scriptPathURI: ads_my_script.py
        type: python
        """
        )
        v = ValidateRuntime(dict_file).validate()
        expected_normalized_dict = {
            "kind": "runtime",
            "spec": {
                "args": ["pos_arg1", "pos_arg2", "--key1", "val1", "--key2", "val2"],
                "conda": {"slug": "mlcpuv1", "type": "service"},
                "env": [
                    {"name": "KEY1", "value": "VALUE1"},
                    {"name": "KEY2", "value": "VALUE2"},
                ],
                "scriptPathURI": "ads_my_script.py",
            },
            "type": "python",
        }
        assert v == expected_normalized_dict

    def test_validate_unknown_kind(self):
        with pytest.raises(TypeError):
            dict_file = yaml.safe_load(
                f"""
                        kind: this_is_invalid
                        type: git
                        spec:
                            url: https://github.com/test_repo/jobs-scripts.git
                            branch: main
                            entrypoint: xgboost_classification_onnx_clean.py
                            entryFunction: main
                            bucketName: test_bucket
                            bucketNamespace: test_namespace
                            bucketPrefix: jobs-git-boot-loader
                        apiVersion: jobs/v1
                    """
            )
            v = ValidatorFactory(dict_file).validate()

    def test_validate_missing_kind(self):
        with pytest.raises(ValueError):
            dict_file = {
                "type": "git",
                "spec": {
                    "url": "https://github.com/test_repo/jobs-scripts.git",
                    "branch": "main",
                    "entrypoint": "xgboost_classification_onnx_clean.py",
                    "entryFunction": "main",
                    "params": '{"data_path": "oci://test_bucket/jobs/etl-out/*.parquet"}',
                    "codeDir": "/home/datascience/app/code",
                    "outputDir": "/home/datascience/output",
                    "bucketName": "test_bucket",
                    "bucketNamespace": "test_namespace",
                    "bucketPrefix": "jobs-git-boot-loader",
                },
                "apiVersion": "jobs/v1",
            }
            response = ValidatorFactory(dict_file).validate()

    def test_validate_invalid_input_dict(self):
        with pytest.raises(ValueError):
            v = ValidatorFactory("test")
