#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import yaml
from ads.jobs.builders.base import Builder


class TestBuilderBase:
    def test_set_spec(self):
        builder = Builder()
        builder.set_spec("k1", "v1")
        assert builder.to_dict()["spec"]["k1"] == "v1"
        builder.set_spec("k2", ["v1"])
        assert builder.to_dict()["spec"]["k2"][0] == "v1"

        builder.set_spec("k3", {"a": "b"})
        assert builder.to_dict()["spec"]["k3"]["a"] == "b"
        builder.set_spec("k3", {"a": "c", "d": "e"})
        assert builder.to_dict()["spec"]["k3"]["a"] == "c"
        assert builder.to_dict()["spec"]["k3"]["d"] == "e"

    def test_to_yaml(self):
        builder = Builder()
        builder.set_spec("k1", "v1")
        builder.set_spec("k2", ["v1"])
        builder.set_spec("k3", {"a": "b"})
        yaml_output = builder.to_yaml()
        assert type(yaml_output) == str
        yaml_dict = yaml.safe_load(yaml_output)
        assert yaml_dict == {
            "kind": "builder",
            "spec": {"k1": "v1", "k2": ["v1"], "k3": {"a": "b"}},
            "type": "builder",
        }

    def test_to_yaml_with_uri(self):
        try:
            f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
            f.close()
            yaml_location = f.name
            builder = Builder()
            builder.set_spec("k1", "v1")
            builder.set_spec("k2", ["v1"])
            builder.set_spec("k3", {"a": "b"})
            yaml_output = builder.to_yaml(uri=yaml_location)
            assert yaml_output == None
            f = open(f.name)
            yaml_content = f.read()
            yaml_dict = yaml.safe_load(yaml_content)
            assert yaml_dict == {
                "kind": "builder",
                "spec": {"k1": "v1", "k2": ["v1"], "k3": {"a": "b"}},
                "type": "builder",
            }
        finally:
            os.unlink(f.name)
