#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from logging import getLogger
from collections import namedtuple
from ads.opctl.config.yaml_parsers import YamlSpecParser

logger = getLogger("ads.yaml")


class DistributedSpecParser(YamlSpecParser):
    def __init__(self, distributed):
        # TODO: validate yamlInput
        self.distributed = distributed

    def parse(self):
        ClusterInfo = namedtuple(
            "ClusterInfo", field_names=["infrastructure", "cluster", "runtime"]
        )
        self.distributed_spec = self.distributed["spec"]
        infrastructure = self.distributed_spec["infrastructure"]
        cluster_def = self.distributed_spec["cluster"]
        cluster = self.parse_cluster(cluster_def)
        runtime = self.parse_runtime(self.distributed_spec.get("runtime"))
        return ClusterInfo(
            infrastructure=infrastructure, cluster=cluster, runtime=runtime
        )

    def parse_cluster(self, cluster_def):
        Cluster = namedtuple(
            "Cluster",
            field_names=[
                "name",
                "type",
                "image",
                "work_dir",
                "config",
                "main",
                "worker",
                "ps",
                "ephemeral",
                "certificate",
            ],
        )
        cluster_spec = cluster_def["spec"]
        name = cluster_spec.get("name")
        cluster_type = cluster_def.get("kind")
        image = cluster_spec.get("image")
        work_dir = cluster_spec.get("workDir")
        ephemeral = cluster_spec.get("ephemeral")
        cluster_default_config = cluster_spec.get("config")
        main = self.parse_main(cluster_spec.get("main"))
        worker = self.parse_worker(cluster_spec.get("worker"))
        ps = self.parse_ps(cluster_spec.get("ps"))
        translated_config = self.translate_config(cluster_default_config)
        certificate = self.parse_certificate(cluster_spec.get("certificate"))
        logger.debug(
            f"Cluster: [name: {name}, type: {cluster_type}, image: {image}, work_dir: {work_dir}, config: {translated_config}, main: {main}, worker: {worker}, ps: {ps}]"
        )
        return Cluster(
            name=name,
            type=cluster_type,
            image=image,
            work_dir=work_dir,
            config=translated_config,
            main=main,
            worker=worker,
            ps=ps,
            ephemeral=ephemeral,
            certificate=certificate,
        )

    def parse_main(self, main):
        Main = namedtuple("Main", field_names=["name", "image", "replicas", "config"])
        main_spec = main
        name = main_spec.get("name")
        replicas = main_spec.get("replicas") or 1
        if replicas > 1:
            logger.warn(
                "`replicas` greater than 1 is currently not supported. This will be default to 1"
            )
        image = main_spec.get("image")
        config = main_spec.get("config")
        translated_config = self.translate_config(config)
        logger.debug(
            f"main: [name: {name}, image: {image}, replicas: {replicas}, config: {translated_config}]"
        )
        return Main(name=name, image=image, replicas=replicas, config=translated_config)

    def parse_worker_params(self, worker_spec):
        name = worker_spec.get("name")
        replicas = worker_spec.get("replicas") or 1
        image = worker_spec.get("image")
        config = worker_spec.get("config")
        translated_config = self.translate_config(config)
        logger.debug(
            f"Worker: [name: {name}, image: {image}, replicas: {replicas}, config: {translated_config}]"
        )
        return name, image, replicas, translated_config

    def parse_worker(self, worker):
        if not worker:
            return None
        Worker = namedtuple("Worker", field_names=["name", "image", "replicas", "config"])
        name, image, replicas, translated_config = self.parse_worker_params(worker)
        logger.debug(
            f"Worker: [name: {name}, image: {image}, replicas: {replicas}, config: {translated_config}]"
        )
        return Worker(name=name, image=image, replicas=replicas, config=translated_config)

    def parse_ps(self, worker):
        if not worker:
            return None
        Ps = namedtuple("PS", field_names=["name", "image", "replicas", "config"])
        name, image, replicas, translated_config = self.parse_worker_params(worker)
        logger.debug(
            f"PS: [name: {name}, image: {image}, replicas: {replicas}, config: {translated_config}]"
        )
        return Ps(name=name, image=image, replicas=replicas, config=translated_config)

    def parse_runtime(self, runtime):
        PythonRuntime = namedtuple(
            "PythonRuntime",
            field_names=[
                "entry_point",
                "args",
                "kwargs",
                "envVars",
                "type",
                "uri",
                "branch",
                "commit",
                "git_secret_id",
                "code_dir",
                "python_path",
            ],
        )
        python_spec = runtime["spec"]
        envVars = {}
        if python_spec.get("env"):
            envVars = {k["name"]: k["value"] for k in python_spec.get("env")}
        return PythonRuntime(
            entry_point=python_spec.get("entryPoint"),
            args=python_spec.get("args"),
            kwargs=python_spec.get("kwargs"),
            envVars=envVars,
            type=runtime.get("type"),
            uri=python_spec.get("uri"),
            branch=python_spec.get("branch"),
            commit=python_spec.get("commit"),
            git_secret_id=python_spec.get("gitSecretId"),
            code_dir=python_spec.get("codeDir"),
            python_path=python_spec.get("pythonPath"),
        )

    def parse_certificate(self, certificate):
        """
        Expected yaml schema:
            cluster:
                spec:
                    certificate:
                        caCert:
                            id: oci.xxxx.<ca_cert_ocid>
                            downloadLocation:  /code/ca.pem
                            cert:
                                id: oci.xxxx.<cert_ocid>
                                certDownloadLocation: /code/cert.pem
                                keyDownloadLocation: /code/key.pem
        """
        if certificate and certificate.get("caCert") and certificate.get("cert"):
            Certificate = namedtuple(
                "Certificate",
                field_names=[
                    "ca_ocid",
                    "ca_download_location",
                    "cert_ocid",
                    "cert_download_location",
                    "key_download_location",
                ],
            )
            ca_ocid = certificate["caCert"]["id"]
            ca_download_location = certificate["caCert"].get(
                "downloadLocation", "ca-cert.pem"
            )
            cert_ocid = certificate["cert"]["id"]
            cert_download_location = certificate["cert"].get(
                "certDownloadLocation", "cert.pem"
            )
            key_download_location = certificate["cert"].get(
                "keyDownloadLocation", "key.pem"
            )
            return Certificate(
                ca_ocid=ca_ocid,
                ca_download_location=ca_download_location,
                cert_ocid=cert_ocid,
                cert_download_location=cert_download_location,
                key_download_location=key_download_location,
            )
