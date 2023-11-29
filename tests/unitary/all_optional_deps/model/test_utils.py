#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.runtime.utils import SchemaValidator
import os
from unittest.mock import patch
import yaml
from cerberus import DocumentError
import pytest


class TestSchemaValidator:
    """TestSchemaValidator class"""

    schema = {
        "NOT_REQUIRED": {"type": "string", "required": False},
        "REQUIRED_NULLABLE": {"type": "string", "required": True, "nullable": True},
        "REQUIRED_NOT_NULLABLE": {
            "type": "string",
            "required": True,
            "nullable": False,
        },
    }

    yaml_dict = {"REQUIRED_NULLABLE": None, "REQUIRED_NOT_NULLABLE": "YES"}
    yaml_dict_fail_required_field = {"REQUIRED_NOT_NULLABLE": "YES"}
    yaml_dict_fail_nullable = {"REQUIRED_NULLABLE": None, "REQUIRED_NOT_NULLABLE": None}
    yaml_dict_fail_wrong_type = {"REQUIRED_NULLABLE": None, "REQUIRED_NOT_NULLABLE": 1}

    @patch("yaml.load", return_value=schema)
    def test__load_schema_validator(self, mock_yaml_load):
        validator = SchemaValidator(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "schema_sample.yaml"
            )
        )
        validator._load_schema_validator()
        assert validator._schema_validator == self.schema

    @patch.object(SchemaValidator, "_load_schema_validator", return_value=schema)
    def test_init(self, mock_load_schema_validator):
        validator = SchemaValidator(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml")
        )
        assert validator.schema_file_path == os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml"
        )
        assert validator._schema_validator == self.schema

    @patch.object(SchemaValidator, "_load_schema_validator", return_value=schema)
    def test_validate(self, mock_load_schema_validator):
        validator = SchemaValidator(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml")
        )
        assert validator.validate(self.yaml_dict)

    @patch.object(SchemaValidator, "_load_schema_validator", return_value=schema)
    def test_validate_fail_required_field(self, mock_load_schema_validator):
        validator = SchemaValidator(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml")
        )
        with pytest.raises(DocumentError):
            validator.validate(self.yaml_dict_fail_required_field)

    @patch.object(SchemaValidator, "_load_schema_validator", return_value=schema)
    def test_validate_fail_nullable(self, mock_load_schema_validator):
        validator = SchemaValidator(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml")
        )
        with pytest.raises(DocumentError):
            validator.validate(self.yaml_dict_fail_nullable)

    @patch.object(SchemaValidator, "_load_schema_validator", return_value=schema)
    def test_validate_fail_wrong_type(self, mock_load_schema_validator):
        validator = SchemaValidator(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_path.yaml")
        )
        with pytest.raises(DocumentError):
            validator.validate(self.yaml_dict_fail_wrong_type)
