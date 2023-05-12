#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import pytest
import yaml

from dataclasses import dataclass, field
from ads.common.serializer import Serializable, DataClassSerializable

try:
    from yaml import CDumper as dumper
    from yaml import CLoader as loader
except:
    from yaml import SafeDumper as dumper
    from yaml import SafeLoader as loader


class Foo(Serializable):
    """A mock class that inherts from Serializable, with implemented abstract methods."""

    def to_dict(self):
        return {"k1": "v1", "k2": "v2"}

    @classmethod
    def from_dict(cls, s):
        return cls()


@dataclass
class Bar(DataClassSerializable):
    """A dataclass that inherts from DataClassSerializable."""

    x: int
    y: int


class TestCommonSerializer:
    """Contains test cases for ads.common.serializer.py"""

    dir = os.path.dirname(os.path.abspath(__file__))

    # test Class Serializable
    def test_not_defined(self):
        """Tests fails to return instance of the class when class is not defined."""

        class Foo(Serializable):
            pass

        with pytest.raises(TypeError):
            f = Foo()

    def test_to_yaml_str(self):
        """Tests returns object serilized as a YAML string."""
        f = Foo()
        x = f.to_yaml()
        assert isinstance(x, str)

    def test_to_and_from_yaml_file(self):
        """Tests creates an object from URI location containing YAML string."""
        f = Foo()
        file_path = "/tmp/xx.yaml"
        f.to_yaml(uri=file_path)
        assert os.path.exists(file_path)

        res = Foo().from_yaml(uri=file_path)
        assert type(res) == Foo

    def test_from_yaml_str(self):
        """Tests creates an object from YAML string provided."""
        yaml_string = yaml.dump({"k1": "v1", "k2": "v2"})
        res = Foo().from_yaml(yaml_string=yaml_string)
        assert type(res) == Foo

    def test_to_json_str(self):
        """Tests returns object serialized as a JSON string."""
        f = Foo()
        x = f.to_json()
        assert isinstance(x, str)

    def test_to_json_with_encoder(self):
        """Tests returns object serialized from a custom data structures as a JSON string."""

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Foo):
                    return [obj.v1, obj.v2]
                return json.JSONEncoder.default(self, obj)

        f = Foo()
        x = f.to_json(encoder=CustomEncoder)
        assert isinstance(x, str)

    def test_to_json_file(self):
        """Tests returns object serialized as a JSON string and saves as JSON file."""
        f = Foo()
        f.to_json(uri="/tmp/xx.json")
        assert os.path.exists("/tmp/xx.json")

    def test_from_json_str(self):
        """Tests creates an object from JSON string provided."""
        json_string = """{"k1": "v1", "k2": "v2"}"""
        f = Foo()
        res = f.from_json(json_string=json_string)
        assert type(res) == Foo

    def test_from_json_with_decoder(self):
        """Tests creates an object from custom decoder."""

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
        """Tests creates an object from URI location containing JSON string."""
        f = Foo()
        file_path = "/tmp/xx.json"
        f.to_json(uri=file_path)
        assert os.path.exists(file_path)

        f2 = Foo()
        res = f2.from_json(uri=file_path)
        assert type(res) == Foo

    def test_from_string_json(self):
        """Tests creates an object from string."""
        json_string = """{"k1": "v1", "k2": "v2"}"""
        res = Foo().from_string(obj_string=json_string)
        assert type(res) == Foo

    def test_from_string_yaml(self):
        """Tests creates an object from URI location containing string."""
        yaml_string = yaml.dump({"k1": "v1", "k2": "v2"})
        res = Foo().from_string(obj_string=yaml_string)
        assert type(res) == Foo

    # test Class DataClassSerializable
    def test_to_dict(self):
        """Tests serializing instance of dataclass into a dictionary."""
        b = Bar(1, 2)
        b_dict = b.to_dict()
        expected_result = {"x": 1, "y": 2}
        assert b_dict == expected_result

    def test_to_dict_fail(self):
        """Tests fails to serialize instance of dataclass into a dictionary."""

        class NotADataClass(DataClassSerializable):
            pass

        obj = NotADataClass()
        with pytest.raises(TypeError):
            obj.to_dict()

    def test_from_dict(self):
        """Tests returning an instance of the class instantiated by the dictionary provided."""
        data = {"x": 1, "y": 2}
        b = Bar.from_dict(data)
        assert type(b) == Bar

        # Tests returning an instance of the class instantiated by the nested dictionary.
        @dataclass
        class CondaDetails(DataClassSerializable):
            slug: str = ""
            env_type: str = ""

        @dataclass
        class MDDetails(DataClassSerializable):
            conda_env: CondaDetails = field(default_factory=CondaDetails)

        nested_data = {"conda_env": {"slug": "", "env_type": ""}}

        obj = MDDetails.from_dict(nested_data)
        assert type(obj.conda_env) == CondaDetails

    def test_from_dict_fail(self):
        """Tests fails to return the instance of the class instantiated by the invalid dictionary."""
        data_list = [1, 2]
        with pytest.raises(TypeError):
            b = Bar.from_dict(data_list)

        data_empty = []
        with pytest.raises(AssertionError):
            b = Bar.from_dict(data_empty)

        data_none = None
        with pytest.raises(AssertionError):
            b = Bar.from_dict(data_none)

    def test_from_dict_nested_dict(self):
        """Tests returning an instance of the class instantiated by the nested dictionary."""

        @dataclass(repr=False)
        class InferenceEnvInfo(DataClassSerializable):
            """Training conda environment info."""

            inference_env_slug: str = ""
            inference_env_type: str = ""
            inference_env_path: str = ""
            inference_python_version: str = ""

        @dataclass(repr=False)
        class TrainingEnvInfo(DataClassSerializable):
            """Training conda environment info."""

            inference_env_slug: str = ""
            inference_env_type: str = ""
            inference_env_path: str = ""
            inference_python_version: str = ""

        @dataclass(repr=False)
        class ModelDeploymentDetails(DataClassSerializable):
            """ModelDeploymentDetails class."""

            inference_conda_env: InferenceEnvInfo = field(
                default_factory=InferenceEnvInfo
            )
            training_conda_env: TrainingEnvInfo = field(default_factory=TrainingEnvInfo)

            def fields(self):
                return ["inference_conda_env", "training_conda_env"]

        @dataclass(repr=False)
        class RuntimeInfo(DataClassSerializable):
            """ModelDeploymentDetails class."""

            deployment: ModelDeploymentDetails = field(
                default_factory=ModelDeploymentDetails
            )

            def fields(self):
                return ["deployment"]

        rt = RuntimeInfo.from_dict(
            {
                "deployment": {
                    "inference_conda_env": {
                        "inference_env_slug": "",
                        "inference_env_type": "",
                        "inference_env_path": "",
                        "inference_python_version": "",
                    },
                    "training_conda_env": {
                        "inference_env_slug": "",
                        "inference_env_type": "",
                        "inference_env_path": "",
                        "inference_python_version": "",
                    },
                }
            }
        )

        assert rt.fields() == ["deployment"]
        assert rt.deployment.fields() == ["inference_conda_env", "training_conda_env"]

    def test_from_dict_field_is_dict(self):
        """Tests returning an instance of the class with dict as field instantiated by the dictionary."""
        from typing import Dict

        @dataclass(repr=False)
        class CondaInfo(DataClassSerializable):
            """Conda Environment Details class."""

            libraries: Dict[str, str] = field(default_factory=dict)

        conda_info = CondaInfo.from_dict(
            {
                "libraries": {
                    "name": "python",
                    "version": "3.7",
                }
            }
        )

        assert isinstance(conda_info, CondaInfo)
        assert isinstance(conda_info.libraries, dict)

    def test__normalize_dict(self):
        obj_dict = {"KEY": "value"}
        lower_obj_dict = DataClassSerializable._normalize_dict(obj_dict)
        assert "key" in lower_obj_dict
