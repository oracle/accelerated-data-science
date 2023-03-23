#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import builtins
import os
import tempfile
from configparser import ConfigParser
from io import StringIO
from unittest import mock

from ads.opctl.distributed.common.abstract_cluster_provider import ClusterProvider
from ads.opctl.cmds import _save_yaml


def yaml_content():
    content = StringIO(
        """
kind: distributed
apiVersion: v1.0
spec:
  infrastructure: # This section maps to Job definition. Does not include environment variables
    kind: infrastructure
    type: dataScienceJob
    apiVersion: v1.0
    spec:
      projectId: pjocid
      compartmentId: ctocid
      displayName: my_distributed_training
      logGroupId: lgrpid
      logId: logid
      subnetId: subID
      shapeName: VM.Standard2.1
      blockStorageSizeGB: 50GB
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: "@default"
      workDir: "Working directory for the cluster"
      ephemeral: True
      name: cluster name
      config:
        env:
          - name: death_timeout
            value: 10
          - name: nprocs
            value: 4
        startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
          - --port 8786
      main:
        image: optional
        name: main-name
        replicas: 1
        config:
          env:
            - name: death_timeout
              value: 20
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
      worker:
        name: worker-name
        image: optional
        replicas: 2 #Name is not decided
        config:
          env:
            - name: death_timeout
              value: 30
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
      ps: #doesn't make sense for dask. Only for UT purpose
        name: ps-name
        image: optional
        replicas: 2
        config:
          env:
            - name: death_timeout
              value: 30
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
  runtime:
    kind: python
    apiVersion: v1.0
    spec:
      entryPoint: "printhello.py"
      args:
        - 500
      kwargs:
      env:
        - name: TEST
          value: "test"
"""
    )
    return content


def yaml_content_2():
    content = StringIO(
        """
kind: distributed
apiVersion: v1.0
spec:
  infrastructure: # This section maps to Job definition. Does not include environment variables
    kind: infrastructure
    type: dataScienceJob
    apiVersion: v1.0
    spec:
      projectId: pjocid
      compartmentId: ctocid
      displayName: my_distributed_training
      logGroupId: lgrpid
      logId: logid
      subnetId: subID
      shapeName: VM.Standard2.1
      blockStorageSizeGB: 50GB
  cluster:
    kind: TENSORFLOW
    apiVersion: v1.0
    spec:
      image: "@default"
      workDir: "Working directory for the cluster"
      ephemeral: True
      name: cluster name
      config:
        env:
          - name: death_timeout
            value: 10
          - name: nprocs
            value: 4
        startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
          - --port 8786
      main:
        image: optional
        name: main-name
        replicas: 1
        config:
          env:
            - name: death_timeout
              value: 20
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
      worker:
        name: worker-name
        image: optional
        replicas: 2 #Name is not decided
        config:
          env:
            - name: death_timeout
              value: 30
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
  runtime:
    kind: python
    apiVersion: v1.0
    spec:
      entryPoint: "printhello.py"
      args:
        - 500
      kwargs:
      env:
        - name: TEST
          value: "test"
"""
    )
    return content


def yaml_content_with_certificate():
    content = StringIO(
        """
kind: distributed
apiVersion: v1.0
spec:
  infrastructure: # This section maps to Job definition. Does not include environment variables
    kind: infrastructure
    type: dataScienceJob
    apiVersion: v1.0
    spec:
      projectId: pjocid
      compartmentId: ctocid
      displayName: my_distributed_training
      logGroupId: lgrpid
      logId: logid
      subnetId: subID
      shapeName: VM.Standard2.1
      blockStorageSizeGB: 50GB
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: "@default"
      workDir: "Working directory for the cluster"
      ephemeral: True
      name: cluster name
      certificate:
        caCert:
            id: oci.xxxx.<ca_cert_ocid>
            downloadLocation: /code/dask-tls-ca-cert.pem
        cert:
            id: oci.xxxx.<cert_ocid>
            certDownloadLocation: /code/dask-tls-cert.pem
            keyDownloadLocation: /code/dask-tls-key.pem
      config:
        env:
          - name: death_timeout
            value: 10
          - name: nprocs
            value: 4
        startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
          - --port 8786
      main:
        image: optional
        name: main-name
        replicas: 1
        config:
          env:
            - name: death_timeout
              value: 20
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
      worker:
        name: worker-name
        image: optional
        replicas: 2 #Name is not decided
        config:
          env:
            - name: death_timeout
              value: 30
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
  runtime:
    kind: python
    apiVersion: v1.0
    spec:
      entryPoint: "printhello.py"
      args:
        - 500
      kwargs:
      env:
        - name: TEST
          value: "test"
"""
    )
    return content


def yaml_content_one_node():
    content = StringIO(
        """
kind: distributed
apiVersion: v1.0
spec:
  infrastructure: # This section maps to Job definition. Does not include environment variables
    kind: infrastructure
    type: dataScienceJob
    apiVersion: v1.0
    spec:
      projectId: pjocid
      compartmentId: ctocid
      displayName: my_distributed_training
      logGroupId: lgrpid
      logId: logid
      subnetId: subID
      shapeName: VM.Standard2.1
      blockStorageSizeGB: 50GB
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: default
      workDir: "Working directory for the cluster"
      ephemeral: True
      name: cluster name
      config:
        env:
          - name: death_timeout
            value: 10
          - name: nprocs
            value: 4
        startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
          - --port 8786
      main:
        image: optional
        name: main-name
        replicas: 1
        config:
          env:
            - name: death_timeout
              value: 20
            - name: nprocs
              value: 4
          startOptions: # Only named args. Will construct OCI__START_ARGS environment variable from list of args as "--key1=value1 --key2=value2..."
            - --port 8786
  runtime:
    kind: python
    apiVersion: v1.0
    spec:
      entryPoint: "printhello.py"
      args:
        - 500
      kwargs:
      env:
        - name: TEST
          value: "test"
"""
    )
    return content


from ads.opctl.config.yaml_parsers import YamlSpecParser
from ads.opctl.distributed.common.cluster_config_helper import (
    ClusterConfigToJobSpecConverter,
)

from ads.opctl.distributed.cmds import (
    update_image,
    increment_tag,
    get_cmd,
    verify_image,
    horovod_cmd,
    pytorch_cmd,
    dask_cmd,
    tensorflow_cmd,
    update_config_image,
)
import yaml


def test_yaml_parsing():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    parsed_output = YamlSpecParser.parse_content(conf)
    print("output:", parsed_output)
    assert parsed_output.cluster.worker.replicas == 2
    assert parsed_output.cluster.name == "cluster name"
    assert parsed_output.cluster.main.name == "main-name"
    assert parsed_output.cluster.worker.name == "worker-name"
    assert parsed_output.cluster.image == "@default"
    assert parsed_output.runtime.entry_point == "printhello.py"
    assert parsed_output.cluster.work_dir == "Working directory for the cluster"


def test_yaml_parsing_with_certificate():
    conf = yaml.load(yaml_content_with_certificate(), yaml.SafeLoader)
    parsed_output = YamlSpecParser.parse_content(conf)
    print("output:", parsed_output)
    assert parsed_output.cluster.worker.replicas == 2
    assert parsed_output.cluster.name == "cluster name"
    assert parsed_output.cluster.main.name == "main-name"
    assert parsed_output.cluster.worker.name == "worker-name"
    assert parsed_output.cluster.image == "@default"
    assert parsed_output.runtime.entry_point == "printhello.py"
    assert parsed_output.cluster.work_dir == "Working directory for the cluster"
    assert parsed_output.cluster.certificate.ca_ocid == "oci.xxxx.<ca_cert_ocid>"
    assert parsed_output.cluster.certificate.cert_ocid == "oci.xxxx.<cert_ocid>"
    assert (
        parsed_output.cluster.certificate.cert_download_location
        == "/code/dask-tls-cert.pem"
    )
    assert (
        parsed_output.cluster.certificate.key_download_location
        == "/code/dask-tls-key.pem"
    )
    assert (
        parsed_output.cluster.certificate.ca_download_location
        == "/code/dask-tls-ca-cert.pem"
    )


def test_job_conf_test():
    global content
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    parsed_output = YamlSpecParser.parse_content(conf)
    print("output:", parsed_output)
    ch = ClusterConfigToJobSpecConverter(parsed_output)
    assert ch.job_def_info() == {
        "name": "cluster name",
        "image": "@default",
        "envVars": {
            "OCI__CLUSTER_TYPE": "DASK",
            "OCI__ENTRY_SCRIPT": "printhello.py",
            "OCI__ENTRY_SCRIPT_ARGS": "500",
            "OCI__EPHEMERAL": "True",
            "OCI__PS_COUNT": "2",
            "OCI__START_ARGS": "--port 8786",
            "OCI__WORKER_COUNT": "2",
            "OCI__WORK_DIR": "Working directory for the cluster",
            "TEST": "test",
            "death_timeout": "10",
            "nprocs": "4",
        },
        "infrastructure": {
            "apiVersion": "v1.0",
            "kind": "infrastructure",
            "spec": {
                "blockStorageSizeGB": "50GB",
                "compartmentId": "ctocid",
                "displayName": "my_distributed_training",
                "logGroupId": "lgrpid",
                "logId": "logid",
                "projectId": "pjocid",
                "shapeName": "VM.Standard2.1",
                "subnetId": "subID",
            },
            "type": "dataScienceJob",
        },
    }
    assert ch.job_run_info("main") == {
        "name": "main-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "MAIN",
            "death_timeout": "20",
            "nprocs": "4",
        },
    }
    assert ch.job_run_info("worker") == {
        "name": "worker-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "WORKER",
            "death_timeout": "30",
            "nprocs": "4",
        },
    }
    assert ch.job_run_info("ps") == {
        "name": "ps-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "PS",
            "death_timeout": "30",
            "nprocs": "4",
        },
    }


def test_job_conf_certificate_test():
    global content
    conf = yaml.load(yaml_content_with_certificate(), yaml.SafeLoader)
    parsed_output = YamlSpecParser.parse_content(conf)
    print("output:", parsed_output)
    ch = ClusterConfigToJobSpecConverter(parsed_output)
    assert ch.job_def_info() == {
        "name": "cluster name",
        "image": "@default",
        "envVars": {
            "OCI__CLUSTER_TYPE": "DASK",
            "OCI__ENTRY_SCRIPT": "printhello.py",
            "OCI__ENTRY_SCRIPT_ARGS": "500",
            "OCI__EPHEMERAL": "True",
            "OCI__START_ARGS": "--port 8786",
            "OCI__WORKER_COUNT": "2",
            "OCI__WORK_DIR": "Working directory for the cluster",
            "OCI__CERTIFICATE_OCID": "oci.xxxx.<cert_ocid>",
            "OCI__CERTIFICATE_AUTHORITY_OCID": "oci.xxxx.<ca_cert_ocid>",
            "OCI__CA_DOWNLOAD_LOCATION": "/code/dask-tls-ca-cert.pem",
            "OCI__CERTIFICATE_DOWNLOAD_LOCATION": "/code/dask-tls-cert.pem",
            "OCI__CERTIFICATE_KEY_DOWNLOAD_LOCATION": "/code/dask-tls-key.pem",
            "TEST": "test",
            "death_timeout": "10",
            "nprocs": "4",
        },
        "infrastructure": {
            "apiVersion": "v1.0",
            "kind": "infrastructure",
            "spec": {
                "blockStorageSizeGB": "50GB",
                "compartmentId": "ctocid",
                "displayName": "my_distributed_training",
                "logGroupId": "lgrpid",
                "logId": "logid",
                "projectId": "pjocid",
                "shapeName": "VM.Standard2.1",
                "subnetId": "subID",
            },
            "type": "dataScienceJob",
        },
    }
    assert ch.job_run_info("main") == {
        "name": "main-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "MAIN",
            "death_timeout": "20",
            "nprocs": "4",
        },
    }
    assert ch.job_run_info("worker") == {
        "name": "worker-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "WORKER",
            "death_timeout": "30",
            "nprocs": "4",
        },
    }


def test_job_conf_one_node_test():
    global content
    conf = yaml.load(yaml_content_one_node(), yaml.SafeLoader)
    parsed_output = YamlSpecParser.parse_content(conf)
    print("output:", parsed_output)
    ch = ClusterConfigToJobSpecConverter(parsed_output)
    assert ch.job_def_info() == {
        "name": "cluster name",
        "image": "default",
        "envVars": {
            "OCI__CLUSTER_TYPE": "DASK",
            "OCI__ENTRY_SCRIPT": "printhello.py",
            "OCI__ENTRY_SCRIPT_ARGS": "500",
            "OCI__EPHEMERAL": "True",
            "OCI__START_ARGS": "--port 8786",
            "OCI__WORKER_COUNT": "0",
            "OCI__WORK_DIR": "Working directory for the cluster",
            "TEST": "test",
            "death_timeout": "10",
            "nprocs": "4",
        },
        "infrastructure": {
            "apiVersion": "v1.0",
            "kind": "infrastructure",
            "spec": {
                "blockStorageSizeGB": "50GB",
                "compartmentId": "ctocid",
                "displayName": "my_distributed_training",
                "logGroupId": "lgrpid",
                "logId": "logid",
                "projectId": "pjocid",
                "shapeName": "VM.Standard2.1",
                "subnetId": "subID",
            },
            "type": "dataScienceJob",
        },
    }
    assert ch.job_run_info("main") == {
        "name": "main-name",
        "envVars": {
            "OCI__START_ARGS": "--port 8786",
            "OCI__MODE": "MAIN",
            "death_timeout": "20",
            "nprocs": "4",
        },
    }

    assert ch.job_run_info("worker") == {}
    assert ch.job_run_info("ps") == {}


def test_update_image_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)

    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "tag", "my_tag")
    ini.set("main", "registry", "my_reg")
    conf = update_image(conf, ini)
    assert conf["spec"]["cluster"]["spec"]["image"] == "my_reg:my_tag"


def test_increment_tag_test():
    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "tag", "mytag")

    ini = increment_tag(ini)
    assert ini.get("main", "tag") == "mytag_v_1"


def test_increment_tag_2_test():
    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "tag", "mytag_v_1")

    ini = increment_tag(ini)
    assert ini.get("main", "tag") == "mytag_v_2"


def test_increment_tag_3_test():
    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "tag", "mytag_v_11")

    ini = increment_tag(ini)
    assert ini.get("main", "tag") == "mytag_v_12"


def test_get_cmd_test():
    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "registry", "my_reg")
    ini.set("main", "tag", "mytag")
    ini.set("main", "dockerfile", "dockerfile")
    ini.set("main", "source_folder", "code")

    cmd = get_cmd(ini)
    assert (
        " ".join(cmd)
        == "docker build --build-arg CODE_DIR=code -t my_reg:mytag -f dockerfile ."
    )


def test_horovod_cmd_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    code_mount = "/target/:/code/"
    oci_key_mount = "/oci/:/home/keys/"
    cmd = horovod_cmd(code_mount, oci_key_mount, conf)

    assert (
        " ".join(cmd)
        == "docker run -v /target/:/code/ -v /oci/:/home/keys/ --env OCI_IAM_TYPE=api_key --rm "
        "--entrypoint /miniconda/envs/env/bin/horovodrun @default --gloo -np 2 -H localhost:2 "
        "/miniconda/envs/env/bin/python printhello.py"
    )


def test_dask_cmd_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    code_mount = "/target/:/code/"
    oci_key_mount = "/oci/:/home/keys/"
    cmd = dask_cmd(code_mount, oci_key_mount, conf)

    assert (
        " ".join(cmd)
        == "docker run -v /target/:/code/ -v /oci/:/home/keys/ --env OCI_IAM_TYPE=api_key --env "
        "SCHEDULER_IP=tcp://127.0.0.1 --rm --entrypoint /bin/sh @default -c (nohup dask-scheduler "
        ">scheduler.log &) && (nohup dask-worker localhost:8786 >worker.log &) && "
        "/miniconda/envs/daskenv/bin/python printhello.py"
    )


def test_pytorch_cmd_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    code_mount = "/target/:/code/"
    oci_key_mount = "/oci/:/home/keys/"
    cmd = pytorch_cmd(code_mount, oci_key_mount, conf)

    assert (
        " ".join(cmd)
        == "docker run -v /target/:/code/ -v /oci/:/home/keys/ --env OCI_IAM_TYPE=api_key --env "
        "WORLD_SIZE=1 --env RANK=0 --env LOCAL_RANK=0 --rm --entrypoint /opt/conda/bin/python "
        "@default printhello.py"
    )


def test_tensorflow_cmd_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    code_mount = "/target/:/code/"
    oci_key_mount = "/oci/:/home/keys/"
    cmd = tensorflow_cmd(code_mount, oci_key_mount, conf)
    assert (
        " ".join(cmd)
        == """docker run -v /target/:/code/ -v /oci/:/home/keys/ --env OCI_IAM_TYPE=api_key --rm --entrypoint /bin/sh @default /etc/datascience/local_test.sh printhello.py"""
    )


def test_tensorflow_cmd_2_test():
    conf = yaml.load(yaml_content_2(), yaml.SafeLoader)
    code_mount = "/target/:/code/"
    oci_key_mount = "/oci/:/home/keys/"
    cmd = tensorflow_cmd(code_mount, oci_key_mount, conf)

    assert (
        " ".join(cmd)
        == """docker run -v /target/:/code/ -v /oci/:/home/keys/ --env OCI_IAM_TYPE=api_key --env TF_CONFIG={"cluster": {"worker": ["localhost:12345"]}, "task": {"type": "worker", "index": 0}} --rm --entrypoint /miniconda/bin/python @default printhello.py"""
    )


class ClusterProviderMock(ClusterProvider):
    def fetch_code(self):
        print("Mocking fetch code")

    def sync(self, loop=True):
        print("Mocking sync")

    def get_oci_auth(self):
        print("Mocking get_oci_auth")

    def setup_configuration(self, config: dict = None):
        print("Mocking setup_configuration")

    @classmethod
    def find_self_ip(cls, authinfo):
        return "127.0.0.1"


def test_profile_cmd_test():
    cp = ClusterProviderMock(
        "MAIN", ephemeral=True, life_span="0h", work_dir="work_dir"
    )
    cmd = cp.profile_cmd()
    assert len(cmd) == 0


def test_profile_cmd_test_2():
    cp = ClusterProviderMock(
        "MAIN", ephemeral=True, life_span="0h", work_dir="work_dir"
    )
    profile_cmd = "nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o /opt/ml/nsight_report -x true"
    os.environ["PROFILE"] = "1"
    os.environ["PROFILE_CMD"] = profile_cmd
    cmd = cp.profile_cmd()
    assert " ".join(cmd) == profile_cmd
    os.environ.pop("PROFILE")
    os.environ.pop("PROFILE_CMD")


def test_get_sync_loc_test():
    cp = ClusterProviderMock(
        "MAIN", ephemeral=True, life_span="0h", work_dir="work_dir"
    )
    os.environ["WORKSPACE"] = "my_bucket"
    os.environ["WORKSPACE_PREFIX"] = "sync/artifacts"
    bckt_name, pfx = cp.get_sync_loc()
    print(bckt_name, pfx)
    assert "my_bucket" == bckt_name
    assert "sync/artifacts" == pfx
    os.environ.pop("WORKSPACE")
    os.environ.pop("WORKSPACE_PREFIX")


def test_get_sync_loc_2_test():
    cp = ClusterProviderMock(
        "MAIN", ephemeral=True, life_span="0h", work_dir="work_dir"
    )
    bckt_name, pfx = cp.get_sync_loc()
    print(bckt_name, pfx)
    assert bckt_name is None
    assert pfx is None


def test_get_sync_loc_3_test():
    cp = ClusterProviderMock(
        "MAIN",
        ephemeral=True,
        life_span="0h",
        work_dir="oci://my_bucket@namespace/my_space/work_dir/",
    )
    bckt_name, pfx = cp.get_sync_loc()
    print(bckt_name, pfx)
    assert "my_bucket" == bckt_name
    assert "my_space/work_dir" == pfx


def test_get_sync_loc_4_test():
    cp = ClusterProviderMock(
        "MAIN",
        ephemeral=True,
        life_span="0h",
        work_dir="oci://my_bucket@namespace/my_space/work_dir",
    )
    bckt_name, pfx = cp.get_sync_loc()
    print(bckt_name, pfx)
    assert "my_bucket" == bckt_name
    assert "my_space/work_dir" == pfx


def test_update_config_image_test():
    conf = yaml.load(yaml_content(), yaml.SafeLoader)
    ini_file = "config.ini"
    ini = ConfigParser(allow_no_value=True)
    ini.add_section("main")
    ini.set("main", "tag", "my_tag")
    ini.set("main", "registry", "my_reg")
    if os.path.exists(ini_file):
        os.remove(ini_file)
    with open(ini_file, "w") as f:
        ini.write(f)
    conf = update_config_image(conf)
    if os.path.exists(ini_file):
        os.remove(ini_file)
    assert conf["spec"]["cluster"]["spec"]["image"] == "my_reg:my_tag"


def test_opctl_run_save_yaml():
    yaml_content = """
  jobId: "ocid1.datasciencejob.oc1.iad.<unique_ocid>"
  mainJobRunId:
    distributed-main: "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
  otherJobRunIds:
  - distributed-worker_0: "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
  - distributed-worker_1: "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
  workDir: oci://test_bucket@test_namespace/distributed
    """
    # Test saving yaml content into a file in temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "info.yaml")
        _save_yaml(yaml_content, job_info=yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            assert f.read() == yaml_content
    # Test saving yaml content into an existing file. T
    # There should be an FileExistsError.
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"abc")
        f.seek(0)
        with mock.patch.object(builtins, "input", lambda _: "N"):
            _save_yaml(yaml_content, job_info=f.name)
            f.seek(0)
            assert f.read() == b"abc"
