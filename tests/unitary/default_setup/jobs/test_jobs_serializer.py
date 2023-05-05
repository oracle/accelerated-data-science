#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import pytest
import yaml

from ads.jobs.serializer import Serializable

try:
    from yaml import CDumper as dumper
    from yaml import CLoader as loader
except:
    from yaml import SafeDumper as dumper
    from yaml import SafeLoader as loader


class Foo(Serializable):
    def to_dict(self):
        return {"k1": "v1", "k2": "v2"}

    @classmethod
    def from_dict(cls, s):
        return cls()


class TestCommonHelper:
    def test_not_defined(self):
        class Foo(Serializable):
            pass

        with pytest.raises(TypeError):
            f = Foo()

    def test_to_yaml_str(self):
        f = Foo()
        x = f.to_yaml()
        assert isinstance(x, str)

    def test_to_and_from_yaml_file(self):
        f = Foo()
        file_path = "/tmp/xx.yaml"
        f.to_yaml(uri=file_path)
        assert os.path.exists(file_path)

        res = Foo().from_yaml(uri=file_path)
        assert type(res) == Foo

    def test_from_yaml_str(self):
        yaml_string = yaml.dump({"k1": "v1", "k2": "v2"})
        res = Foo().from_yaml(yaml_string=yaml_string)
        assert type(res) == Foo

    def test_to_json_str(self):
        f = Foo()
        x = f.to_json()
        assert isinstance(x, str)

    def test_to_json_with_encoder(self):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Foo):
                    return [obj.v1, obj.v2]
                return json.JSONEncoder.default(self, obj)

        f = Foo()
        x = f.to_json(encoder=CustomEncoder)
        assert isinstance(x, str)

    def test_to_json_file(self):
        f = Foo()
        f.to_json(uri="/tmp/xx.json")
        assert os.path.exists("/tmp/xx.json")

    def test_from_json_str(self):
        json_string = """{"k1": "v1", "k2": "v2"}"""
        f = Foo()
        res = f.from_json(json_string=json_string)
        assert type(res) == Foo

    def test_from_json_with_decoder(self):
        class CustomDecoder(json.JSONDecoder):
            def default(self, str_input):
                if type(str_input) == str:
                    dct = yaml.load(str_input, Loader=loader)
                    return Foo(dct["k1"], dct["k2"])
                return dct

        json_string = """{"k1": "v1", "k2": "v2"}"""
        res = Foo().from_json(json_string=json_string, decoder=CustomDecoder)
        assert type(res) == Foo

    def test_from_json_file(self):
        f = Foo()
        file_path = "/tmp/xx.json"
        f.to_json(uri=file_path)
        assert os.path.exists(file_path)

        f2 = Foo()
        res = f2.from_json(uri=file_path)
        assert type(res) == Foo

    def test_from_string_json(self):
        json_string = """{"k1": "v1", "k2": "v2"}"""
        res = Foo().from_string(obj_string=json_string)
        assert type(res) == Foo

    def test_from_string_yaml(self):
        yaml_string = yaml.dump({"k1": "v1", "k2": "v2"})
        res = Foo().from_string(obj_string=yaml_string)
        assert type(res) == Foo
