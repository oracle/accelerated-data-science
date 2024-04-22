#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import tempfile
import os
import json
from mock import ANY, patch
import pytest
from pathlib import Path
from unittest import SkipTest

try:
    from ads.opctl.backend.local import LocalBackend, CondaPackNotFound
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


class TestLocalBackend:
    def test_init_vscode_conda(self):
        with tempfile.TemporaryDirectory() as td:
            backend = LocalBackend(
                {
                    "execution": {
                        "image": "image-name",
                        "use_conda": True,
                        "source_folder": td,
                        "oci_config": "~/.oci/config",
                        "conda_pack_folder": "~/conda",
                        "env_vars": {},
                        "volumes": {},
                    }
                }
            )
            with pytest.raises(ValueError):
                with pytest.raises(RuntimeError):
                    backend.init_vscode_container()
                    with open(os.path.join(td, ".devcontainer.json")) as f:
                        content = json.load(f)
                    assert content == {
                        "image": "image-name",
                        "extensions": ["ms-python.python"],
                        "mounts": [
                            f"source={os.path.expanduser('~/.oci')},target=/home/datascience/.oci,type=bind",
                            f"source={os.path.expanduser('~/conda')},target=/opt/conda/envs,type=bind",
                        ],
                        "workspaceMount": f"source={td},target=/home/datascience/{os.path.basename(td)},type=bind",
                        "workspaceFolder": "/home/datascience",
                        "containerEnv": {},
                        "name": "ml-job-dev-env",
                        "postCreateCommand": "conda init bash && source ~/.bashrc",
                    }

    def test_init_vscode_image(self):
        with tempfile.TemporaryDirectory() as td:
            backend = LocalBackend(
                {
                    "execution": {
                        "image": "image-name",
                        "oci_config": "~/.oci/config",
                        "conda_pack_folder": "~/conda",
                        "env_vars": {},
                        "volumes": {},
                    }
                }
            )
            with pytest.raises(RuntimeError):
                backend.init_vscode_container()
                with open(".devcontainer.json") as f:
                    content = json.load(f)
                assert content == {
                    "image": "image-name",
                    "mounts": [],
                    "extensions": ["ms-python.python"],
                    "containerEnv": {},
                }

    @property
    def config(self):
        return {
            "execution": {
                "backend": "local",
                "use_conda": True,
                "debug": False,
                "env_var": ["TEST_ENV=test_env"],
                "oci_config": "~/.oci/config",
                "oci_profile": "BOAT",
                "command": "-n hello-world -c ~/.oci/config -p BOAT",
                "image": "ml-job",
                "env_vars": {"TEST_ENV": "test_env"},
                "job_name": "hello-world",
            },
        }

    def test_check_conda(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "conda", "slug"))
            config = self.config
            config["execution"]["conda_pack_folder"] = os.path.join(td, "conda")
            volume, _ = LocalBackend(
                config
            )._check_conda_pack_and_install_if_applicable("slug", {}, {})
            assert os.path.join(td, "conda", "slug") in volume
            with pytest.raises(CondaPackNotFound):
                LocalBackend(config)._check_conda_pack_and_install_if_applicable(
                    "nonexist_slug", {}, {}
                )
            config["execution"][
                "conda_uri"
            ] = "oci://bucket@namespace/path/to/some_slug"
            LocalBackend(config)._check_conda_pack_and_install_if_applicable(
                "slug", {}, {}
            )

    def test_mount_source_folder(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            config = self.config
            config["execution"]["source_folder"] = os.path.join(td, "src")
            volume = LocalBackend(config)._mount_source_folder_if_exists({})
            assert os.path.join(td, "src") in volume
            config["execution"]["source_folder"] = os.path.join(td, "nonexist_src")
            with pytest.raises(FileNotFoundError):
                LocalBackend(config)._mount_source_folder_if_exists({})

    @patch("ads.opctl.backend.local.run_command")
    @patch("ads.opctl.backend.local.LocalBackend._activate_conda_env_and_run")
    def test_run_with_conda(self, run_func, mock_run_cmd, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "conda", "slug"))
            os.makedirs(os.path.join(td, "src"))
            config = self.config
            config["execution"]["operator_name"] = "hello-world"
            config["execution"]["conda_pack_folder"] = os.path.join(td, "conda")
            config["execution"]["conda_slug"] = "slug"
            config["execution"]["source_folder"] = os.path.join(td, "src")
            config["execution"]["auth"] = "api_key"
            backend = LocalBackend(config)
            backend._run_with_conda_pack({})
            run_func.assert_called_with(
                "ml-job",
                "slug",
                "python /etc/datascience/operators/run.py -n hello-world -c ~/.oci/config -p BOAT",
                {
                    os.path.join(td, "conda", "slug"): {"bind": "/opt/conda/envs/slug"},
                    os.path.join(td, "src"): {"bind": "/etc/datascience/operators/src"},
                },
                {"TEST_ENV": "test_env"},
            )

            monkeypatch.setenv("NB_SESSION_OCID", "abcde")

            backend._run_with_conda_pack({})
            import ads

            mock_run_cmd.assert_called_with(
                f"python {os.path.join(ads.__path__[0], 'opctl', 'operators', 'run.py')} -n hello-world -c ~/.oci/config -p BOAT",
                shell=True,
            )

            config["execution"]["auth"] = "resource_principal"
            backend = LocalBackend(config)
            backend._run_with_conda_pack({})
            mock_run_cmd.assert_called_with(
                f"python {os.path.join(ads.__path__[0], 'opctl', 'operators', 'run.py')} -r -n hello-world -c ~/.oci/config -p BOAT",
                shell=True,
            )

            monkeypatch.delenv("NB_SESSION_OCID", raising=True)

        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "conda", "pyspark"))
            os.makedirs(os.path.join(td, "src"))
            Path(os.path.join(td, "src", "main.py")).touch()
            with open(
                os.path.join(td, "conda", "pyspark", "spark-defaults.conf"), "w"
            ) as f:
                f.write(
                    """
spark.driver.extraClassPath=/home/datascience/spark_conf_dir/bc-fips-1.0.2.jar:/home/datascience/spark_conf_dir/bcpkix-fips-1.0.2.jar:/home/datascience/spark_conf_dir/oci-hdfs-full-3.3.1.0.3.1.jar:/home/datascience/spark_conf_dir/ojdbc8.jar:/home/datascience/spark_conf_dir/oraclepki.jar:/home/datascience/spark_conf_dir/osdt_cert.jar:/home/datascience/spark_conf_dir/osdt_core.jar:/home/datascience/spark_conf_dir/ucp.jar
spark.executor.extraClassPath=/home/datascience/spark_conf_dir/bc-fips-1.0.2.jar:/home/datascience/spark_conf_dir/bcpkix-fips-1.0.2.jar:/home/datascience/spark_conf_dir/oci-hdfs-full-3.3.1.0.3.1.jar:/home/datascience/spark_conf_dir/ojdbc8.jar:/home/datascience/spark_conf_dir/oraclepki.jar:/home/datascience/spark_conf_dir/osdt_cert.jar:/home/datascience/spark_conf_dir/osdt_core.jar:/home/datascience/spark_conf_dir/ucp.jar
spark.driver.extraLibraryPath=/home/datascience/spark_conf_dir/hadoop-3.3.0/lib/native
spark.ui.enabled=false
spark.driver.bindAddress=127.0.0.1
spark.driver.host=127.0.0.1
                """
                )
            Path(os.path.join(td, "src", "main.py")).touch()

            config = self.config
            config["execution"]["entrypoint"] = "main.py"
            config["execution"]["conda_pack_folder"] = os.path.join(td, "conda")
            config["execution"]["conda_slug"] = "pyspark"
            config["execution"]["source_folder"] = os.path.join(td, "src")
            config["execution"]["oci_profile"] = "DEFAULT"
            config["execution"]["auth"] = "api_key"
            config["execution"][
                "command"
            ] = "-n hello-world -c ~/.oci/config -p DEFAULT"
            backend = LocalBackend(config)
            backend._run_with_conda_pack({})
            run_func.assert_called_with(
                "ml-job",
                "pyspark",
                "python /etc/datascience/operators/src/main.py -n hello-world -c ~/.oci/config -p DEFAULT",
                {
                    os.path.join(td, "conda", "pyspark"): {
                        "bind": "/opt/conda/envs/pyspark"
                    },
                    os.path.join(td, "src"): {"bind": "/etc/datascience/operators/src"},
                },
                {"TEST_ENV": "test_env", "SPARK_CONF_DIR": "/opt/conda/envs/pyspark"},
            )
            assert os.path.exists(os.path.join(td, "conda", "pyspark", "core-site.xml"))

    @patch("ads.opctl.backend.local.run_container")
    def test_run_with_image(self, run_container):
        config = self.config
        config["execution"]["operator_name"] = "hello-world"
        config["execution"]["image"] = "image-name"
        config["execution"]["volumes"] = {"src": {"bind": "dst"}}
        LocalBackend(config)._run_with_image({})
        run_container.assert_called_with(
            "image-name",
            {"src": {"bind": "dst"}},
            {"TEST_ENV": "test_env"},
            "python /etc/datascience/operators/run.py -n hello-world -c ~/.oci/config -p BOAT",
            None,
        )
        config["execution"].pop("operator_name")
        config["execution"]["entrypoint"] = "python main.py"
        LocalBackend(config)._run_with_image({})
        run_container.assert_called_with(
            "image-name",
            {"src": {"bind": "dst"}},
            {"TEST_ENV": "test_env"},
            "-n hello-world -c ~/.oci/config -p BOAT",
            "python main.py",
        )

    @patch("ads.opctl.backend.local.get_docker_client")
    @patch("ads.opctl.backend.local.run_container", return_value=7)
    def test_run_that_fails(self, run_container, get_docker_client):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "conda", "slug"))
            os.makedirs(os.path.join(td, "src"))
            Path(os.path.join(td, "src", "test.py")).touch()
            config = self.config
            config["execution"]["conda_pack_folder"] = os.path.join(td, "conda")
            config["execution"]["conda_slug"] = "slug"
            config["execution"]["source_folder"] = os.path.join(td, "src")
            config["execution"]["auth"] = "api_key"
            config["execution"]["entrypoint"] = "test.py"
            backend = LocalBackend(config)
            with pytest.raises(
                RuntimeError,
                match="^Job did not complete successfully. Exit code: 7(.*)$",
            ):
                backend.run()
            run_container.assert_called_with(
                "ml-job",
                ANY,
                {"TEST_ENV": "test_env", "conda_slug": "slug"},
                command="bash /etc/datascience/entryscript.sh",
            )
