#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile

import pytest

import ads.opctl
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.config.utils import read_from_ini
from ads.opctl.constants import (
    DEFAULT_OCI_CONFIG_FILE,
)
from ads.opctl.utils import list_ads_operators


class TestConfigMerger:
    def test_merge_configs_with_cmd_args(self):
        config = {
            "spec": {"name": "a", "list": ["a", "b"], "bool": False},
            "type": "t1",
            "version": "v1",
        }
        m = ConfigMerger(config)
        m._merge_config_with_cmd_args(
            {
                "name": "b",
                "type": "t2",
                "backend": "local",
                "list": [],
                "version": None,
                "bool": None,
                "extra": (),
            },
        )
        assert (
            m.config["spec"]["name"] == "b"
            and m.config["type"] == "t2"
            and m.config["execution"]["backend"] == "local"
            and m.config["spec"]["list"] == ["a", "b"]
            and m.config["spec"]["bool"] is False
            and m.config["version"] == "v1"
            and "extra" not in m.config["execution"]
        )

    def test_read_from_ini(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "conf.ini"), "w") as f:
                f.write(
                    """
[A]
a_Property: aA
                """
                )
            parser = read_from_ini(os.path.join(td, "conf.ini"))
            assert parser["A"] == {"a_Property": "aA"}

    def test_fill_config_from_conf(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "conf.ini"), "w") as f:
                f.write(
                    """
[A]
name: a


[B]
name: b
value: 2
                """
                )
            config = {
                "spec": {"name": "$name", "value": "$value"},
                "execution": {"conf_file": os.path.join(td, "conf.ini")},
            }
            config["execution"]["conf_profile"] = "A"
            m = ConfigMerger(config)
            m._fill_config_from_conf()
            assert m.config["spec"]["name"] == "a" and not m.config["spec"]["value"]
            config["execution"]["conf_profile"] = "B"
            m = ConfigMerger(config)
            m._fill_config_from_conf()
            assert m.config["spec"]["name"] == "b" and m.config["spec"]["value"] == 2
            config["spec"]["value"] = 3
            m = ConfigMerger(config)
            m._fill_config_from_conf()
            assert m.config["spec"]["name"] == "b" and m.config["spec"]["value"] == 3

    def test_fill_config_with_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "config.ini"), "w") as f:
                f.write(
                    """
[OCI]
oci_profile = PROFILE

[CONDA]
conda_pack_folder = ~/condapack
conda_pack_os_prefix = oci://bucket@namespace/path
                """
                )
            with open(os.path.join(td, "ml_job_config.ini"), "w") as f:
                f.write(
                    """
[PROFILE]
compartment_id: oci.compartmentid.abcd
project_id: oci.projectid.abcd
conda_pack_os_prefix: oci://bucket@namespace/path2

[PROFILE2]
compartment_id: oci.compartmentid.efgh
project_id: oci.projectid.efgh

[RESOURCE_PRINCIPAL]
compartment_id: oci.compartmentid.hijk
project_id: oci.projectid.hijk
conda_pack_os_prefix: oci://bucket@namespace/path3
                """
                )
            m = ConfigMerger({"execution": {"auth": "api_key"}})
            m._fill_config_with_defaults(td)
            assert m.config == {
                "execution": {
                    "auth": "api_key",
                    "oci_config": DEFAULT_OCI_CONFIG_FILE,
                    "oci_profile": "PROFILE",
                    "conda_pack_folder": "~/condapack",
                    "conda_pack_os_prefix": "oci://bucket@namespace/path2",
                },
                "infrastructure": {
                    "compartment_id": "oci.compartmentid.abcd",
                    "project_id": "oci.projectid.abcd",
                },
            }
            m = ConfigMerger(
                {"execution": {"oci_profile": "PROFILE2", "auth": "api_key"}}
            )
            m._fill_config_with_defaults(td)
            assert m.config == {
                "execution": {
                    "auth": "api_key",
                    "oci_config": DEFAULT_OCI_CONFIG_FILE,
                    "oci_profile": "PROFILE2",
                    "conda_pack_folder": "~/condapack",
                    "conda_pack_os_prefix": "oci://bucket@namespace/path",
                },
                "infrastructure": {
                    "compartment_id": "oci.compartmentid.efgh",
                    "project_id": "oci.projectid.efgh",
                },
            }
            m = ConfigMerger({"execution": {"auth": "resource_principal"}})
            m._fill_config_with_defaults(td)
            assert m.config == {
                "execution": {
                    "auth": "resource_principal",
                    "oci_config": DEFAULT_OCI_CONFIG_FILE,
                    "oci_profile": None,
                    "conda_pack_folder": "~/condapack",
                    "conda_pack_os_prefix": "oci://bucket@namespace/path3",
                },
                "infrastructure": {
                    "compartment_id": "oci.compartmentid.hijk",
                    "project_id": "oci.projectid.hijk",
                },
            }

    def test_config_flex_shape_details(self):
        config_one = {
            "execution": {
                "backend": "job",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "PROFILE",
                "conda_pack_folder": "~/condapack",
                "conda_pack_os_prefix": "oci://bucket@namespace/path2",
            },
            "infrastructure": {
                "compartment_id": "oci.compartmentid.abcd",
                "project_id": "oci.projectid.abcd",
                "shape_name": "VM.Standard.E2.4"
            },
        }

        m = ConfigMerger(config_one)
        m._config_flex_shape_details()

        assert m.config == {
            "execution": {
                "backend": "job",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "PROFILE",
                "conda_pack_folder": "~/condapack",
                "conda_pack_os_prefix": "oci://bucket@namespace/path2",
            },
            "infrastructure": {
                "compartment_id": "oci.compartmentid.abcd",
                "project_id": "oci.projectid.abcd",
                "shape_name": "VM.Standard.E2.4"
            },
        }
        
        config_one["infrastructure"]["shape_name"] = "VM.Standard.E3.Flex"
        m = ConfigMerger(config_one)

        with pytest.raises(
            ValueError, 
            match="Parameters `ocpus` and `memory_in_gbs` must be provided for using flex shape. "
                    "Call `ads opctl config` to specify."
        ):
            m._config_flex_shape_details()    

        config_one["infrastructure"]["ocpus"] = 2
        config_one["infrastructure"]["memory_in_gbs"] = 24
        m = ConfigMerger(config_one)
        m._config_flex_shape_details()

        assert m.config == {
            "execution": {
                "backend": "job",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "PROFILE",
                "conda_pack_folder": "~/condapack",
                "conda_pack_os_prefix": "oci://bucket@namespace/path2",
            },
            "infrastructure": {
                "compartment_id": "oci.compartmentid.abcd",
                "project_id": "oci.projectid.abcd",
                "shape_name": "VM.Standard.E3.Flex",
                "shape_config_details": {
                    "ocpus": 2,
                    "memory_in_gbs": 24
                }
            },
        }

        config_two = {
            "execution": {
                "backend": "dataflow",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "PROFILE",
                "conda_pack_folder": "~/condapack",
                "conda_pack_os_prefix": "oci://bucket@namespace/path2",
            },
            "infrastructure": {
                "compartment_id": "oci.compartmentid.abcd",
                "project_id": "oci.projectid.abcd",
                "executor_shape": "VM.Standard.E3.Flex",
                "driver_shape": "VM.Standard.E3.Flex"
            },
        }

        m = ConfigMerger(config_two)

        with pytest.raises(
            ValueError, 
            match="Parameters driver_shape_memory_in_gbs must be provided for using flex shape. "
                    "Call `ads opctl config` to specify."
        ):
            m._config_flex_shape_details()


        config_two["infrastructure"]["driver_shape_memory_in_gbs"] = 36
        config_two["infrastructure"]["driver_shape_ocpus"] = 4
        config_two["infrastructure"]["executor_shape_memory_in_gbs"] = 48
        config_two["infrastructure"]["executor_shape_ocpus"] = 5

        m = ConfigMerger(config_two)
        m._config_flex_shape_details()
        assert m.config == {
            "execution": {
                "backend": "dataflow",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "PROFILE",
                "conda_pack_folder": "~/condapack",
                "conda_pack_os_prefix": "oci://bucket@namespace/path2",
            },
            "infrastructure": {
                "compartment_id": "oci.compartmentid.abcd",
                "project_id": "oci.projectid.abcd",
                "executor_shape": "VM.Standard.E3.Flex",
                "executor_shape_config": {
                    "ocpus": 5,
                    "memory_in_gbs": 48
                },
                "driver_shape": "VM.Standard.E3.Flex",
                "driver_shape_config": {
                    "ocpus": 4,
                    "memory_in_gbs": 36
                }
            },
        }

class TestConfigResolver:
    def test_resolve_operator_name(self):
        config = {"name": "name1", "execution": {"operator_name": "name2"}}
        r = ConfigResolver(config)
        r._resolve_operator_name()
        assert r.config["execution"]["operator_name"] == "name2"
        config = {"name": "name1", "execution": {}}
        r = ConfigResolver(config)
        r._resolve_operator_name()
        assert r.config["execution"]["operator_name"] == "name1"

    def test_resolve_source_folder_path(self):
        config = {"execution": {"operator_name": "hello-world"}}
        r = ConfigResolver(config)
        r._resolve_source_folder_path()
        assert r.config["execution"]["source_folder"] == os.path.join(
            ads.opctl.__path__[0], "operators", "hello_world"
        )

    def test_resolve_command(self):
        config = {
            "execution": {
                "operator_name": "hello-world",
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "backend": "local",
            },
            "spec": {"name": "abc"},
        }
        r = ConfigResolver(config)
        r._resolve_command()
        assert (
            r.config["execution"]["command"]
            == "-n hello-world -c ~/.oci/config -p DEFAULT -s eyJuYW1lIjogImFiYyJ9"
        )

    def test_resolve_conda(self):
        ads_operators = list_ads_operators()
        config = {
            "execution": {
                "conda_slug": "database_p37_v1",
                "conda_pack_folder": "~/conda",
            }
        }
        r = ConfigResolver(config)
        r._resolve_conda()
        assert (
            r.config["execution"]["conda_slug"] == "database_p37_v1"
            and r.config["execution"]["conda_type"] == "service"
        )
        config = {"execution": {"conda_uri": "oci://bkt@ns/path/to/published_pack"}}
        r = ConfigResolver(config)
        r._resolve_conda()
        assert (
            r.config["execution"]["conda_slug"] == "published_pack"
            and r.config["execution"]["conda_type"] == "published"
        )
        config = {"execution": {"operator_name": "hello-world", "use_conda": True}}
        r = ConfigResolver(config)
        r._resolve_conda()
        assert (
            r.config["execution"]["conda_slug"] == "dataexpl_p37_cpu_v2"
            and r.config["execution"]["conda_type"] == "service"
        )

    @pytest.mark.skip(
        "This test is doing api call to oci, has to be mocked or moved to 'integration' tests folder."
    )
    def test_resolve_image_name(self):
        config = {
            "execution": {
                "operator_name": "hello-world",
                "auth": "api_key",
                "oci_config": DEFAULT_OCI_CONFIG_FILE,
                "oci_profile": "DEFAULT",
            }
        }
        r = ConfigResolver(config)
        r._resolve_image_name()
        assert os.path.basename(r.config["execution"]["image"]) == "hello-world"

    def test_resolve_env_vars(self):
        config = {"execution": {"env_vars": {"ev1": "v1"}, "env_var": ["ev2=v2"]}}
        r = ConfigResolver(config)
        r._resolve_env_vars()
        assert r.config["execution"]["env_vars"] == {"ev1": "v1", "ev2": "v2"}

    def test_resolve_job_name(self):
        config = {"execution": {"job_name": "abc"}}
        r = ConfigResolver(config)
        r._resolve_job_name()
        assert r.config["execution"]["job_name"] == "abc"
        config = {"execution": {"operator_name": "abc", "entrypoint": "/path/to/file"}}
        r = ConfigResolver(config)
        r._resolve_job_name()
        assert r.config["execution"]["job_name"] == "abc"

    def test_resolve_config(self):
        config = {"execution": {}}
        r = ConfigResolver(config)
        with pytest.raises(ValueError):
            r.process()
        config = {
            "execution": {
                "conda_slug": "xxxx",
                "image": "yyyyy",
                "conda_pack_folder": "~/conda",
            }
        }
        r = ConfigResolver(config)
        with pytest.raises(ValueError):
            r.process()
