#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
from xml.dom import minidom
from xml.etree.ElementTree import Element
from xml.etree import ElementTree as et

from ads.common.auth import AuthType, create_signer
from ads.opctl.config.utils import read_from_ini
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.utils import get_oci_region, is_in_notebook_session
from ads.opctl.constants import DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR


def core_site(auth, oci_config, overwrite, oci_profile):

    if is_in_notebook_session():
        if not os.path.exists(
            os.path.join(DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR, "spark-defaults.conf")
        ):
            raise RuntimeError("Pyspark conda pack is not installed.")
        core_site_loc = os.path.join(
            DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR, "core-site.xml"
        )
    else:
        if "CONDA_PREFIX" not in os.environ or not os.path.exists(
            os.path.join(os.environ["CONDA_PREFIX"], "spark-defaults.conf")
        ):
            raise RuntimeError("Please run this inside a Pyspark conda environment.")
        core_site_loc = os.path.join(os.environ["CONDA_PREFIX"], "core-site.xml")

    if os.path.exists(core_site_loc) and not overwrite:
        print(f"{core_site_loc} already exists. Please use `--overwrite` option.")
        return

    p = ConfigProcessor().step(
        ConfigMerger,
        oci_config=oci_config,
        oci_profile=oci_profile,
        auth=auth,
    )
    exec_config = p.config["execution"]
    oci_config = exec_config["oci_config"]
    oci_profile = exec_config["oci_profile"]
    auth = exec_config["auth"]

    properties = generate_core_site_properties(auth, oci_config, oci_profile)
    xmlstr = generate_core_site_properties_str(properties)

    with open(core_site_loc, "w") as f:
        f.write(xmlstr)

    print(f"The core-site.xml is being written to {core_site_loc}")


def generate_core_site_properties(
    authentication,
    oci_config=None,
    oci_profile=None,
):

    region = os.getenv("NB_REGION", None)
    if oci_config and oci_profile:
        region = region or get_oci_region(
            create_signer(AuthType.API_KEY, oci_config, oci_profile)
        )

    if authentication == "api_key":
        if not oci_config or not oci_profile:
            raise ValueError(
                "oci-config and oci-profile must be provided when using api_key."
            )
        oci_config_path = os.path.abspath(os.path.expanduser(oci_config))

        if not os.path.exists(oci_config_path):
            raise RuntimeError(
                f"The oci config path {oci_config_path} does not exist. Please refer to the getting-started.ipynb on generating oci API keys."
            )

        config = read_from_ini(oci_config_path)
        if oci_profile not in config:
            raise RuntimeError(
                f"The configuration profile {oci_profile} does not exist in {oci_config_path}"
            )
        details_dict = config[oci_profile]

        for key in ["user", "fingerprint", "key_file", "tenancy", "region"]:
            if key not in details_dict.keys():
                raise ValueError(
                    f"{key} not found in oci config. Is {oci_config_path} is corrupt?"
                )

        # setup region
        details_dict["region"] = region

        return [
            (
                "fs.oci.client.hostname",
                f"https://objectstorage.{details_dict['region']}.oraclecloud.com",
            ),
            ("fs.oci.client.auth.tenantId", details_dict["tenancy"]),
            ("fs.oci.client.auth.userId", details_dict["user"]),
            ("fs.oci.client.auth.fingerprint", details_dict["fingerprint"]),
            (
                "fs.oci.client.auth.pemfilepath",
                os.path.expanduser(details_dict["key_file"]),
            ),
        ]

    else:
        return [
            (
                "fs.oci.client.hostname",
                f"https://objectstorage.{region}.oraclecloud.com",
            ),
            (
                "fs.oci.client.custom.authenticator",
                "com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator",
            ),
        ]


def generate_core_site_properties_str(properties):
    def core_site_property(name, value):
        elem = Element("property")
        child_name = Element("name")
        child_name.text = name
        child_value = Element("value")
        child_value.text = value
        elem.append(child_name)
        elem.append(child_value)
        return elem

    elem = Element("configuration")
    for (name, value) in properties:
        child = core_site_property(name, value)
        elem.append(child)

    return (
        minidom.parseString(et.tostring(elem)).childNodes[0].toprettyxml(indent="   ")
    )
