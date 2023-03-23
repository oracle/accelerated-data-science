#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model metadata module. Includes tests for:
 - ModelTaxonomyMetadataItem
 - ModelCustomMetadataItem
 - ModelTaxonomyMetadata
 - ModelCustomMetadata
"""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
import numpy as np
from ads.model.model_metadata import (
    _METADATA_EMPTY_VALUE,
    METADATA_SIZE_LIMIT,
    METADATA_VALUE_LENGTH_LIMIT,
    METADATA_DESCRIPTION_LENGTH_LIMIT,
    MetadataCustomCategory,
    Framework,
    MetadataSizeTooLarge,
    MetadataValueTooLong,
    MetadataDescriptionTooLong,
    ModelCustomMetadata,
    ModelCustomMetadataItem,
    ModelTaxonomyMetadata,
    ModelTaxonomyMetadataItem,
    MetadataTaxonomyKeys,
    UseCaseType,
)
from oci.data_science.models import Metadata as OciMetadataItem

try:
    from yaml import CDumper as dumper
except:
    from yaml import Dumper as dumper


class TestModelTaxonomyMetadataItem:
    """Unittests for ModelTaxonomyMetadataItem class."""

    def setup_method(self):
        self.test_key = MetadataTaxonomyKeys.USE_CASE_TYPE
        self.test_value = UseCaseType.BINARY_CLASSIFICATION
        self.test_key1 = MetadataTaxonomyKeys.FRAMEWORK
        self.test_value1 = Framework.XGBOOST
        self.test_item = ModelTaxonomyMetadataItem(
            key=self.test_key, value=self.test_value
        )

    def test_key_fail(self):
        """Ensures item key fails in case of wrong input data."""

        # test non-string input for key
        with pytest.raises(TypeError) as execinfo:
            item = ModelTaxonomyMetadataItem(
                key=MagicMock(),
                value=self.test_value,
            )
        assert str(execinfo.value) == "The key must be a string."

        # test empty input for key
        with pytest.raises(ValueError) as execinfo:
            item = ModelTaxonomyMetadataItem(key="", value=self.test_value)
        assert str(execinfo.value) == "The key cannot be empty."

        # test key should be immutable
        with pytest.raises(ValueError) as execinfo:
            item = ModelCustomMetadataItem(key=self.test_key, value=self.test_value)
            item.key = "new key"
        assert (
            str(execinfo.value) == "The key field is immutable and cannot be changed."
        )

    def test_key_success(self):
        """Tests setting item key."""
        item = ModelCustomMetadataItem(key=self.test_key, value=self.test_value)
        item.key == self.test_key

    def test_value_fail(self):
        """Ensures setting item value fails in case of not JSON serializeble input value."""
        with pytest.raises(ValueError) as exc:
            self.test_item.value = MagicMock()
        assert str(exc.value) == (
            f"An error occurred in attempt to serialize the value of {self.test_item.key} to JSON. "
            "The value must be JSON serializable."
        )

    @pytest.mark.parametrize(
        "test_value", [0.5, {"a": 0, "b": 1}, [1, 2, 3], (1, 2, 3), None]
    )
    def test_value_success(self, test_value):
        """Tests setting item vaue."""
        self.test_item.value = test_value
        assert self.test_item.value == test_value

    def test_update(self):
        """Tests updating metadata item value."""

        self.test_item.update(value=UseCaseType.CLUSTERING)
        assert self.test_item.value == UseCaseType.CLUSTERING

    def test_to_dict(self):
        """Tests serializing model metadata item to dictionary."""
        item_dict = self.test_item.to_dict()
        expected_result = {
            "key": self.test_item.key,
            "value": self.test_item.value,
        }
        assert item_dict == expected_result

    def test_to_yaml(self):
        """Tests serializing model metadata item to YAML."""
        item_yaml = self.test_item.to_yaml()
        assert item_yaml == yaml.dump(self.test_item.to_dict(), Dumper=dumper)

    def test_size(self):
        """Tests calculating size of the model metadata in bytes."""
        assert self.test_item.size() == len(
            json.dumps(self.test_item.to_dict()).encode("utf-16")
        )

    def test_reset(self):
        """Tests reseting model metadata item."""
        self.test_item.reset()
        assert self.test_item.value == None

    def test_to_json(self):
        """Tests serializing metadata item into a JSON."""
        expected_result = json.dumps(self.test_item.to_dict())
        test_result = self.test_item.to_json()
        assert expected_result == test_result

    def test_validate(self):
        """Tests validating metadata item."""

        # UseCaseType
        test_item = ModelTaxonomyMetadataItem(
            key=MetadataTaxonomyKeys.USE_CASE_TYPE,
            value=UseCaseType.CLUSTERING,
        )
        assert test_item.validate() == True

        test_item.value = "invalid value"
        with pytest.raises(ValueError) as exc:
            test_item.validate()
        assert (
            str(exc.value)
            == f"Invalid value of `UseCaseType`. Choose from {UseCaseType.values()}."
        )

        # Framework
        test_item = ModelTaxonomyMetadataItem(
            key=MetadataTaxonomyKeys.FRAMEWORK, value=Framework.XGBOOST
        )
        assert test_item.validate() == True

        test_item.value = "invalid value"
        with pytest.raises(ValueError) as exc:
            test_item.validate()
        assert (
            str(exc.value)
            == f"Invalid value of `Framework`. Choose from {Framework.values()}."
        )

        # Any other key
        test_item = ModelTaxonomyMetadataItem(
            key=MetadataTaxonomyKeys.ALGORITHM, value="any value"
        )
        assert test_item.validate() == True

    @pytest.mark.parametrize(
        "test_value, expected_value",
        [
            ("1", "1"),
            (1, "1"),
            (1.5, "1.5"),
            ({"key": "test_key"}, json.dumps({"key": "test_key"})),
            (None, None),
            ("", ""),
        ],
    )
    def test__to_oci_metadata(self, test_value, expected_value):
        """Tests converting metadata item to OCI metadata item."""
        # case with non empty string value
        test_metadata_item = ModelTaxonomyMetadataItem(
            key=self.test_key,
            value=test_value,
        )

        expected_oci_metadata_item = OciMetadataItem(
            key=self.test_key,
            value=expected_value,
        )
        result_oci_metadata_item = test_metadata_item._to_oci_metadata()
        assert expected_oci_metadata_item == result_oci_metadata_item
        assert result_oci_metadata_item.value == expected_value

    @pytest.mark.parametrize(
        "test_value, expected_value",
        [
            (None, None),
            ("", ""),
            ("1", 1),
            ("str", "str"),
            (json.dumps({"key": "test_key"}), {"key": "test_key"}),
            (1.2, 1.2),
            ({"key": "test_key"}, {"key": "test_key"}),
        ],
    )
    def test__from_oci_metadata(self, test_value, expected_value):
        """Tests creating a new metadata item from the OCI metadata item."""
        test_oci_metadata_item = OciMetadataItem(
            key=self.test_key,
            value=test_value,
        )
        expected_model_metadata_item = ModelTaxonomyMetadataItem(
            key=self.test_key, value=expected_value
        )
        result_metadata_item = ModelTaxonomyMetadataItem._from_oci_metadata(
            test_oci_metadata_item
        )
        assert expected_model_metadata_item == result_metadata_item
        assert result_metadata_item.value == expected_value


class TestModelCustomMetadataItem:
    """Unit tests for ModelCustomMetadata class."""

    KEY = "SlugName"
    VALUE = "pyspark30_p37_cpu_v1"
    DESCRIPTION = "The slug name which was uesd to train the model."
    CATEGORY = MetadataCustomCategory.PERFORMANCE
    ITEM = ModelCustomMetadataItem(
        key=KEY,
        value=VALUE,
        description=DESCRIPTION,
        category=CATEGORY,
    )

    def test_item(self):
        # test init and get metadata's info
        item = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        assert item.key == self.KEY
        assert item.value == self.VALUE
        assert item.description == self.DESCRIPTION
        assert item.category == self.CATEGORY

    def test_item_description(self):
        # test replace description
        self.ITEM.description = "new description"
        assert self.ITEM.description == "new description"

    def test_item_category(self):
        # test replace category
        self.ITEM.category = MetadataCustomCategory.TRAINING_PROFILE
        assert self.ITEM.category == MetadataCustomCategory.TRAINING_PROFILE

        # test None for category
        self.ITEM.category = None
        assert self.ITEM.category == None

        # test non-string input for category
        with pytest.raises(TypeError) as execinfo:
            self.ITEM.category = 0.5
        assert str(execinfo.value) == (
            f"Invalid category type for the {self.ITEM.key}."
            "The category must be a string."
        )

        with pytest.raises(ValueError) as execinfo:
            self.ITEM.category = "Not supported category"
        assert str(execinfo.value) == (
            f"Invalid category value for the {self.ITEM.key}. "
            f"Choose from {MetadataCustomCategory.values()}."
        )

    def test_item_update(self):
        self.ITEM = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )

        # test update
        self.ITEM.update(
            value=self.VALUE, description=self.DESCRIPTION, category=self.CATEGORY
        )
        assert self.ITEM.value == self.VALUE
        assert self.ITEM.description == self.DESCRIPTION
        assert self.ITEM.category == self.CATEGORY

    def test_item_to_dict(self):
        item = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        # test to dictionary format
        item_dict = item.to_dict()
        assert item_dict["key"] == self.KEY
        assert item_dict["value"] == self.VALUE
        assert item_dict["description"] == self.DESCRIPTION
        assert item_dict["category"] == self.CATEGORY

    def test_item_to_yaml(self):
        item = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        # test to yaml format
        item_yaml = item.to_yaml()
        assert item_yaml == yaml.dump(item.to_dict(), Dumper=dumper)

    def test_item_size(self):
        item = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        # test size
        item.size() == len(json.dumps(item.to_dict()).encode("utf-16"))

    def test_item_reset(self):
        # test reset
        self.ITEM.reset()
        assert self.ITEM.value == None
        assert self.ITEM.description == None
        assert self.ITEM.category == None

    def test_to_json(self):
        """Tests serializing metadata item into a JSON."""
        expected_result = json.dumps(self.ITEM.to_dict())
        test_result = self.ITEM.to_json()
        assert expected_result == test_result

    def test_to_json_file_fail(self):
        """Ensures saving metadata to JSON fails in case of wrong input parameters."""
        with pytest.raises(TypeError) as err:
            self.ITEM.to_json_file(file_path={"not_valid_path": ""})
        assert str(err.value) == "File path must be a string."

        with pytest.raises(ValueError) as err:
            self.ITEM.to_json_file(file_path=None)
        assert str(err.value) == "File path must be specified."

    def test_to_json_file_success(self):
        """Ensures saving metadata to JSON passes in case of valid input parameters."""
        mock_file_name = "test_file_name.json"
        mock_file_path = "oci://bucket-name@namespace/"
        mock_storage_options = {"config": {"test": "value"}}
        expected_result = self.ITEM.value
        open_mock = mock_open()

        # Tests saving without file name.
        with patch("fsspec.open", open_mock, create=True):
            self.ITEM.to_json_file(
                file_path=mock_file_path, storage_options=mock_storage_options
            )
        open_mock.assert_called_with(
            mock_file_path + "SlugName.json", mode="w", **mock_storage_options
        )
        open_mock.return_value.write.assert_called_with(json.dumps(expected_result))

        # Tests saving with spectific file name
        mock_file_path = mock_file_path + mock_file_name
        with patch("fsspec.open", open_mock, create=True):
            self.ITEM.to_json_file(
                file_path=mock_file_path,
                storage_options=mock_storage_options,
            )
        open_mock.assert_called_with(mock_file_path, mode="w", **mock_storage_options)
        open_mock.return_value.write.assert_called_with(json.dumps(expected_result))

    @pytest.mark.parametrize(
        "test_value, expected_value",
        [
            ("1", "1"),
            (1, "1"),
            (1.5, "1.5"),
            ({"key": "test_key"}, json.dumps({"key": "test_key"})),
            (None, _METADATA_EMPTY_VALUE),
            ("", _METADATA_EMPTY_VALUE),
            ({"key": np.NaN}, json.dumps({"key": np.NaN}).replace("NaN", "null")),
        ],
    )
    def test__to_oci_metadata(self, test_value, expected_value):
        """Tests converting metadata item to OCI metadata item."""
        # case with non empty string value
        test_metadata_item = ModelCustomMetadataItem(
            key=self.KEY,
            value=test_value,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )

        expected_oci_metadata_item = OciMetadataItem(
            key=self.KEY,
            value=expected_value,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        result_oci_metadata_item = test_metadata_item._to_oci_metadata()
        assert expected_oci_metadata_item == result_oci_metadata_item
        assert result_oci_metadata_item.value == expected_value

    @pytest.mark.parametrize(
        "test_value, expected_value",
        [
            (None, None),
            ("", ""),
            ("1", 1),
            ("str", "str"),
            (json.dumps({"key": "test_key"}), {"key": "test_key"}),
            (1.2, 1.2),
            ({"key": "test_key"}, {"key": "test_key"}),
        ],
    )
    def test__from_oci_metadata(self, test_value, expected_value):
        """Tests creating a new metadata item from the OCI metadata item."""
        test_oci_metadata_item = OciMetadataItem(
            key=self.KEY,
            value=test_value,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        expected_model_metadata_item = ModelCustomMetadataItem(
            key=self.KEY,
            value=expected_value,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        result_metadata_item = ModelCustomMetadataItem._from_oci_metadata(
            test_oci_metadata_item
        )
        assert expected_model_metadata_item == result_metadata_item
        assert result_metadata_item.value == expected_value

    def test_validate(self):
        """Tests validating metadata item."""
        metadata_item = ModelCustomMetadataItem(
            key=self.KEY,
            value=self.VALUE,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        assert metadata_item.validate() == True

        # not supported category
        metadata_item._category = "not supported category"
        with pytest.raises(ValueError) as exc:
            metadata_item.validate()
        assert str(exc.value) == (
            f"Invalid category value for the {self.KEY}. "
            f"Choose from {MetadataCustomCategory.values()}."
        )

        # exceeds value length limit
        metadata_item = ModelCustomMetadataItem(
            key=self.KEY,
            value=[1] * METADATA_VALUE_LENGTH_LIMIT,
            description=self.DESCRIPTION,
            category=self.CATEGORY,
        )
        with pytest.raises(MetadataValueTooLong) as exc:
            metadata_item.validate()

        # exceeds description length limit
        metadata_item = ModelCustomMetadataItem(
            key=self.KEY,
            value="test",
            description="ab" * METADATA_DESCRIPTION_LENGTH_LIMIT,
            category=self.CATEGORY,
        )
        with pytest.raises(MetadataDescriptionTooLong) as exc:
            metadata_item.validate()


class TestModelCustomMetadata:
    """Unit tests for ModelCustomMetadata class."""

    performance_item = ModelCustomMetadataItem(
        key="SlugName",
        value="pyspark30_p37_cpu_v1",
        description="The slug name which was uesd to train the model.",
        category=MetadataCustomCategory.PERFORMANCE,
    )

    user_defined_item = ModelCustomMetadataItem(
        key="My Own Meta",
        value="My own Meta",
        description="This is my own meta",
        category=MetadataCustomCategory.OTHER,
    )

    dict_item = ModelCustomMetadataItem(
        key="My Meta With Dictionary", value={"key1": {"key2": 11, "key3": "value3"}}
    )

    empty_value_item = ModelCustomMetadataItem(
        key="My Meta With Empty Value", value=None
    )

    def test__add(self):
        # test init model metadata
        metadata_custom = ModelCustomMetadata()

        # test add wrong type model metadata item
        with pytest.raises(TypeError) as execinfo:
            metadata_custom._add("")
        assert (
            execinfo.value.args[0]
            == "Argument must be an instance of the class ModelCustomMetadataItem."
        )
        metadata_custom._add(self.performance_item)

        # test get by key
        assert metadata_custom.get("SlugName") == self.performance_item
        # test add exist model metadata item without replace
        with pytest.raises(ValueError) as execinfo:
            metadata_custom._add(self.performance_item)
        assert (
            execinfo.value.args[0]
            == "The metadata item with key SlugName is already registered. Use replace=True to overwrite."
        )

        # test get empty key
        with pytest.raises(ValueError) as execinfo:
            metadata_custom.get("")
        assert execinfo.value.args[0] == "The key must not be an empty string."

        # test get non-exist key
        with pytest.raises(ValueError) as execinfo:
            metadata_custom.get("not exist")
        assert execinfo.value.args[0] == "The metadata with not exist not found."

        # test add mutiple new model metadata
        metadata_custom._add_many([self.user_defined_item, self.dict_item])
        assert metadata_custom.get("My Own Meta") == self.user_defined_item
        assert metadata_custom.get("My Meta With Dictionary") == self.dict_item
        with pytest.raises(TypeError) as execinfo:
            metadata_custom._add_many(self.user_defined_item)
        assert (
            execinfo.value.args[0] == "Argument must be a list of model metadata items."
        )

        # test add by replace
        new_performance_item = ModelCustomMetadataItem(
            key="SlugName",
            value="new value",
            description="new description",
            category=MetadataCustomCategory.PERFORMANCE,
        )

        metadata_custom._add_many([new_performance_item], replace=True)
        assert metadata_custom.get("SlugName") == new_performance_item

    def test_remove(self):
        # test remove model metadata item
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        metadata_custom.remove("SlugName")
        with pytest.raises(ValueError) as execinfo:
            metadata_custom.get("SlugName")
        assert execinfo.value.args[0] == "The metadata with SlugName not found."

    def test_clear(self):
        # test remove all model metadata item
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        metadata_custom.clear()
        with pytest.raises(ValueError) as execinfo:
            metadata_custom.get("SlugName")
        assert execinfo.value.args[0] == "The metadata with SlugName not found."

    def test_keys(self):
        # test if return all keys
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add_many(
            [self.performance_item, self.user_defined_item, self.dict_item]
        )
        keys = metadata_custom.keys
        for item in [self.performance_item, self.user_defined_item, self.dict_item]:
            assert item.key in keys

    def test_is_empty(self):
        # test if metadata is empty
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        assert metadata_custom.isempty() == False
        metadata_custom.clear()
        assert metadata_custom.isempty() == True

    def test_reset(self):
        # test reset model metadata
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)

        assert metadata_custom.get("SlugName").value == self.performance_item.value

        metadata_custom.reset()
        assert metadata_custom.get("SlugName").value == None

    def test_to_dataframe(self):
        # test to_dataframe model metadata
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        df = metadata_custom.to_dataframe()
        assert df["Key"].iloc[0] == self.performance_item.key
        assert df["Value"].iloc[0] == self.performance_item.value
        assert df["Description"].iloc[0] == self.performance_item.description
        assert df["Category"].iloc[0] == self.performance_item.category

    def test_size(self):
        # test check size of model metadata
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        assert metadata_custom.size() == self.performance_item.size()

    def test_validate_size(self):
        # test check size validation of model metadata
        metadata_custom = ModelCustomMetadata()
        large_item = ModelCustomMetadataItem(
            key="Large",
            value=[[1] * METADATA_SIZE_LIMIT],
            description="This a very large item",
            category=MetadataCustomCategory.OTHER,
        )
        metadata_custom._add(large_item)
        with pytest.raises(MetadataSizeTooLarge):
            assert metadata_custom.validate_size()

    def test_validate(self):
        # test check validation of model metadata
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)

        # test for valid metadata
        assert metadata_custom.validate() == True

        # test for too large metadata
        large_item = ModelCustomMetadataItem(
            key="Large",
            value=[[1] * METADATA_VALUE_LENGTH_LIMIT],
            description="This a very large item",
            category=MetadataCustomCategory.OTHER,
        )
        metadata_custom._add(large_item)
        with pytest.raises(ValueError) as execinfo:
            assert metadata_custom.validate()
        assert execinfo.value.args[0].startswith("The custom metadata value")
        self.performance_item.value = "pyspark30_p37_cpu_v1"

    def test_to_dict(self):
        # test to dictionary format
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        assert metadata_custom.to_dict()["data"] == [self.performance_item.to_dict()]

    def test_to_yaml(self):
        # test to yaml format
        metadata_custom = ModelCustomMetadata()
        metadata_custom._add(self.performance_item)
        assert metadata_custom.to_yaml() == yaml.dump(
            metadata_custom.to_dict(), Dumper=dumper
        )

    def test__to_oci_metadata(self):
        """Tests converting model custom metadata to a list of OCI metadata objects."""
        test_metadata_custom = ModelCustomMetadata()
        test_metadata_custom._add(self.performance_item)
        expected_oci_metadata_item = self.performance_item._to_oci_metadata()
        test_result = test_metadata_custom._to_oci_metadata()
        assert len(test_result) == 1
        assert test_result[0] == expected_oci_metadata_item
        assert test_result[0].value == expected_oci_metadata_item.value

        # test empty value
        test_metadata_custom = ModelCustomMetadata()
        test_metadata_custom._add(self.empty_value_item)
        expected_oci_metadata_item = self.empty_value_item._to_oci_metadata()
        test_result = test_metadata_custom._to_oci_metadata()
        assert len(test_result) == 1
        assert test_result[0] == expected_oci_metadata_item
        assert test_result[0].value == _METADATA_EMPTY_VALUE

    def test__from_oci_metadata(self):
        """Tests converting from list of oci metadata to a list of model custom metadata object."""
        test_oci_metadata_item = self.performance_item._to_oci_metadata()
        test_metadata_custom = ModelCustomMetadata._from_oci_metadata(
            [test_oci_metadata_item]
        )
        assert len(test_metadata_custom) == 1
        assert test_metadata_custom[self.performance_item.key] == self.performance_item

    def test_add_fail(self):
        """Ensures adding new model metadata items fail in case of wrong input parameters."""
        test_metadata_custom = ModelCustomMetadata()
        test_metadata_custom._add(self.performance_item)

        with pytest.raises(ValueError) as exc:
            test_metadata_custom.add(key=None, value="test_value")
        assert str(exc.value) == "The key cannot be empty."

        with pytest.raises(ValueError) as exc:
            test_metadata_custom.add(key="test_key", value=None)
        assert str(exc.value) == "The value cannot be empty."

        with pytest.raises(TypeError) as exc:
            test_metadata_custom.add(key=MagicMock(), value="test_value")
        assert str(exc.value) == "The key must be a string."

        with pytest.raises(TypeError) as exc:
            test_metadata_custom.add(
                key="key", value="test_value", category=MagicMock()
            )
        assert str(exc.value) == "The category must be a string."

        with pytest.raises(ValueError) as exc:
            test_metadata_custom.add(
                key="key", value="test_value", category="not supported category"
            )
        assert str(exc.value) == (
            f"Invalid category value. "
            f"Choose from {MetadataCustomCategory.values()}."
        )

        with pytest.raises(ValueError) as exc:
            test_metadata_custom.add(key="key", value=MagicMock())
        assert str(exc.value) == (
            f"An error occurred in attempt to serialize the value of `key` to JSON. "
            "The value must be JSON serializable."
        )

        with pytest.raises(TypeError) as exc:
            test_metadata_custom.add(
                key="key", value="test_value", description=MagicMock()
            )
        assert str(exc.value) == "The description must be a string."

        with pytest.raises(MetadataValueTooLong):
            test_metadata_custom.add(key="key", value=[1] * METADATA_VALUE_LENGTH_LIMIT)

        with pytest.raises(MetadataDescriptionTooLong):
            test_metadata_custom.add(
                key="key",
                value="test_value",
                description="ab" * METADATA_DESCRIPTION_LENGTH_LIMIT,
            )

        with pytest.raises(ValueError) as exc:
            test_metadata_custom.add(
                key=self.performance_item.key, value=self.performance_item.value
            )
        assert str(exc.value) == (
            f"The metadata item with key {self.performance_item.key} is already registered. "
            "Use replace=True to overwrite."
        )

    def test_add_success(self):
        """Tests adding a new model metadata item."""
        test_metadata_custom = ModelCustomMetadata()
        test_metadata_custom._add(self.performance_item)

        test_metadata_custom.add(
            key=self.performance_item.key,
            value=self.performance_item.value,
            description=self.performance_item.description,
            category=MetadataCustomCategory.PERFORMANCE,
            replace=True,
        )
        assert len(test_metadata_custom) == 1
        assert (
            test_metadata_custom[self.performance_item.key].value
            == self.performance_item.value
        )
        assert (
            test_metadata_custom[self.performance_item.key].description
            == self.performance_item.description
        )
        assert (
            test_metadata_custom[self.performance_item.key].category
            == MetadataCustomCategory.PERFORMANCE
        )

        test_metadata_custom.add(
            key=self.user_defined_item.key,
            value=self.user_defined_item.value,
            description=self.user_defined_item.description,
        )
        assert len(test_metadata_custom) == 2
        assert (
            test_metadata_custom[self.user_defined_item.key].value
            == self.user_defined_item.value
        )
        assert (
            test_metadata_custom[self.user_defined_item.key].description
            == self.user_defined_item.description
        )
        assert (
            test_metadata_custom[self.user_defined_item.key].category
            == MetadataCustomCategory.OTHER
        )

    def test_set_training_and_validation_dataset(self):
        metadata_custom = ModelCustomMetadata()
        metadata_custom.set_training_data(
            path="oci://bucket_name@namespace/train_data_filename",
            data_size="(200,100)",
        )
        metadata_custom.set_validation_data(
            path="oci://bucket_name@namespace/validation_data_filename",
            data_size="(100,100)",
        )
        assert (
            metadata_custom["TrainingDataset"].value
            == "oci://bucket_name@namespace/train_data_filename"
        )
        assert metadata_custom["TrainingDatasetSize"].value == "(200,100)"

        assert (
            metadata_custom["ValidationDataset"].value
            == "oci://bucket_name@namespace/validation_data_filename"
        )
        assert metadata_custom["ValidationDatasetSize"].value == "(100,100)"


class TestModelTaxonomyMetadata:
    """Unit tests for ModelTaxonomyMetadata class."""

    def test_metadata_taxonomy(self):
        metadata_taxonomy = ModelTaxonomyMetadata()
        # test set and get metadata
        assert metadata_taxonomy.get("Algorithm").value == None
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        assert metadata_taxonomy.get("Algorithm").value == "ensemble"
        # test update metadata
        metadata_taxonomy.get("Algorithm").update(value="ensemble1")
        assert metadata_taxonomy.get("Algorithm").value == "ensemble1"

    def test_model_taxonomy_reset(self):
        # test reset all metadata items
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        metadata_taxonomy.reset()
        assert metadata_taxonomy.get("Algorithm").value == None

    def test_model_taxonomy_to_dataframe(self):
        # test converting metadata to a data frame format
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        df = metadata_taxonomy.to_dataframe()

        assert df.loc[df["Key"] == "Algorithm"]["Value"][0] == "ensemble"

    def test_model_taxonomy_size(self):
        # test calculating model metadata size
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        assert metadata_taxonomy.size() == sum(
            item.size() for item in metadata_taxonomy._items
        )

    def test_model_taxonomy_validate(self):
        # test check validation of model metadata
        metadata_taxonomy = ModelTaxonomyMetadata()
        assert metadata_taxonomy.validate() == True
        metadata_taxonomy.get(
            MetadataTaxonomyKeys.USE_CASE_TYPE
        ).value = "not valid value"
        with pytest.raises(ValueError) as execinfo:
            assert metadata_taxonomy.validate()
        assert execinfo.value.args[0].startswith("Invalid value of")

    def test_model_taxonomy_to_dict(self):
        # test to dictionary format
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        assert metadata_taxonomy.to_dict()["data"] == [
            item.to_dict() for item in metadata_taxonomy._items
        ]

    def test_model_taxonomy_to_yaml(self):
        # test to yaml format
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy.get("Algorithm").value = "ensemble"
        assert metadata_taxonomy.to_yaml() == yaml.dump(
            metadata_taxonomy.to_dict(), Dumper=dumper
        )

    @patch.object(ModelTaxonomyMetadataItem, "_to_oci_metadata", return_value=None)
    def test__to_oci_metadata(self, mock_to_oci_metadata):
        """Tests converting model taxonomy metadata to a list of OCI metadata objects."""
        metadata_taxonomy = ModelTaxonomyMetadata()
        test_result = metadata_taxonomy._to_oci_metadata()
        assert len(test_result) == len(metadata_taxonomy)
        mock_to_oci_metadata.assert_called()

    def test__from_oci_metadata(self):
        """Tests converting from list of oci metadata to a list of model taxonomy metadata object."""
        test_oci_metadata_list = [
            OciMetadataItem(key=MetadataTaxonomyKeys.FRAMEWORK, value="test_framework"),
            OciMetadataItem(
                key=MetadataTaxonomyKeys.FRAMEWORK_VERSION,
                value="test_framework_version",
            ),
        ]
        metadata_taxonomy = ModelTaxonomyMetadata._from_oci_metadata(
            test_oci_metadata_list
        )
        assert (
            metadata_taxonomy[MetadataTaxonomyKeys.FRAMEWORK].value
            == test_oci_metadata_list[0].value
        )
        assert (
            metadata_taxonomy[MetadataTaxonomyKeys.FRAMEWORK_VERSION].value
            == test_oci_metadata_list[1].value
        )

    def test_to_dataframe(self):
        # test to_dataframe model metadata
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy[MetadataTaxonomyKeys.ALGORITHM].value = "algorithm"
        df = metadata_taxonomy.to_dataframe()
        assert df["Key"].iloc[0] == MetadataTaxonomyKeys.ALGORITHM
        assert df["Value"].iloc[0] == "algorithm"

    def test__populate_from_map(self):
        """Tests populating metadata information from map."""
        test_map = {
            MetadataTaxonomyKeys.USE_CASE_TYPE: UseCaseType.BINARY_CLASSIFICATION,
            MetadataTaxonomyKeys.FRAMEWORK: Framework.SCIKIT_LEARN,
        }
        metadata_taxonomy = ModelTaxonomyMetadata()
        metadata_taxonomy._populate_from_map(test_map)
        assert (
            metadata_taxonomy[MetadataTaxonomyKeys.USE_CASE_TYPE].value
            == UseCaseType.BINARY_CLASSIFICATION
        )
        assert (
            metadata_taxonomy[MetadataTaxonomyKeys.FRAMEWORK].value
            == Framework.SCIKIT_LEARN
        )

    def test_to_json_file_fail(self):
        """Ensures saving metadata to JSON fails in case of wrong input parameters."""
        metadata_taxonomy = ModelTaxonomyMetadata()
        with pytest.raises(TypeError) as err:
            metadata_taxonomy.to_json_file(file_path={"not_valid_path": ""})
        assert str(err.value) == "File path must be a string."

        with pytest.raises(ValueError) as err:
            metadata_taxonomy.to_json_file(file_path=None)
        assert str(err.value) == "File path must be specified."

    def test_to_json_file_success(self):
        """Ensures saving metadata to JSON passes in case of valid input parameters."""
        mock_file_name = "test_file_name.json"
        mock_file_path = "oci://bucket-name@namespace/"
        mock_storage_options = {"config": {"test": "value"}}

        metadata_taxonomy = ModelTaxonomyMetadata()
        open_mock = mock_open()

        # Tests saving without file name.
        with patch("fsspec.open", open_mock, create=True):
            metadata_taxonomy.to_json_file(
                file_path=mock_file_path, storage_options=mock_storage_options
            )
        open_mock.assert_called_with(
            mock_file_path + "ModelTaxonomyMetadata.json",
            mode="w",
            **mock_storage_options,
        )
        open_mock.return_value.write.assert_called_with(metadata_taxonomy.to_json())

        # Tests saving with spectific file name
        mock_file_path = mock_file_path + mock_file_name
        with patch("fsspec.open", open_mock, create=True):
            metadata_taxonomy.to_json_file(
                file_path=mock_file_path,
                storage_options=mock_storage_options,
            )
        open_mock.assert_called_with(mock_file_path, mode="w", **mock_storage_options)
        open_mock.return_value.write.assert_called_with(metadata_taxonomy.to_json())
