#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Tuple

import ads.dataset.factory as factory
import fsspec
import git
import oci.data_science.models
import pandas as pd
import yaml
from ads.common import logger
from ads.common.error import ChangesNotCommitted
from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.serializer import DataClassSerializable
from ads.common.object_storage_details import ObjectStorageDetails
from oci.util import to_dict

try:
    from yaml import CDumper as dumper
except:
    from yaml import Dumper as dumper


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ADS")

METADATA_SIZE_LIMIT = 32000
METADATA_VALUE_LENGTH_LIMIT = 255
METADATA_DESCRIPTION_LENGTH_LIMIT = 255
_METADATA_EMPTY_VALUE = "NA"
CURRENT_WORKING_DIR = "."


class MetadataSizeTooLarge(ValueError):
    """Maximum allowed size for model metadata has been exceeded.
    See https://docs.oracle.com/en-us/iaas/data-science/using/models_saving_catalog.htm for more details.
    """

    def __init__(self, size: int):
        super().__init__(
            f"The metadata is `{size}` bytes and exceeds "
            f"the size limit of `{METADATA_SIZE_LIMIT}` bytes. "
            "Reduce the metadata size."
        )


class MetadataValueTooLong(ValueError):
    """Maximum allowed length of metadata value has been exceeded.
    See https://docs.oracle.com/en-us/iaas/data-science/using/models_saving_catalog.htm for more details.
    """

    def __init__(self, key: str, length: int):
        super().__init__(
            f"The custom metadata value of `{key}` is `{length}` characters and exceeds "
            f"the length limit of `{METADATA_VALUE_LENGTH_LIMIT}` characters."
        )


class MetadataDescriptionTooLong(ValueError):
    """Maximum allowed length of metadata description has been exceeded.
    See https://docs.oracle.com/en-us/iaas/data-science/using/models_saving_catalog.htm for more details.
    """

    def __init__(self, key: str, length: int):
        super().__init__(
            f"The custom metadata description of `{key}` is `{length}` characters and exceeds "
            f"the length limit of `{METADATA_DESCRIPTION_LENGTH_LIMIT}` characters."
        )


class MetadataCustomPrintColumns(str, metaclass=ExtendedEnumMeta):
    KEY = "Key"
    VALUE = "Value"
    DESCRIPTION = "Description"
    CATEGORY = "Category"


class MetadataTaxonomyPrintColumns(str, metaclass=ExtendedEnumMeta):
    KEY = "Key"
    VALUE = "Value"


class MetadataTaxonomyKeys(str, metaclass=ExtendedEnumMeta):
    USE_CASE_TYPE = "UseCaseType"
    FRAMEWORK = "Framework"
    FRAMEWORK_VERSION = "FrameworkVersion"
    ALGORITHM = "Algorithm"
    HYPERPARAMETERS = "Hyperparameters"
    ARTIFACT_TEST_RESULT = "ArtifactTestResults"


class MetadataCustomKeys(str, metaclass=ExtendedEnumMeta):
    SLUG_NAME = "SlugName"
    CONDA_ENVIRONMENT = "CondaEnvironment"
    CONDA_ENVIRONMENT_PATH = "CondaEnvironmentPath"
    ENVIRONMENT_TYPE = "EnvironmentType"
    MODEL_ARTIFACTS = "ModelArtifacts"
    TRAINING_DATASET = "TrainingDataset"
    VALIDATION_DATASET = "ValidationDataset"
    MODEL_SERIALIZATION_FORMAT = "ModelSerializationFormat"
    TRAINING_DATASET_SIZE = "TrainingDatasetSize"
    VALIDATION_DATASET_SIZE = "ValidationDatasetSize"
    TRAINING_DATASET_NUMBER_OF_ROWS = "TrainingDatasetNumberOfRows"
    TRAINING_DATASET_NUMBER_OF_COLS = "TrainingDatasetNumberOfCols"
    VALIDATION_DATASET_NUMBER_OF_ROWS = "ValidationDatasetNumberOfRows"
    VALIDATION_DATASET_NUMBER_OF_COLS = "ValidationDataSetNumberOfCols"
    CLIENT_LIBRARY = "ClientLibrary"
    MODEL_FILE_NAME = "ModelFileName"


class MetadataCustomCategory(str, metaclass=ExtendedEnumMeta):
    PERFORMANCE = "Performance"
    TRAINING_PROFILE = "Training Profile"
    TRAINING_AND_VALIDATION_DATASETS = "Training and Validation Datasets"
    TRAINING_ENV = "Training Environment"
    OTHER = "Other"


class UseCaseType(str, metaclass=ExtendedEnumMeta):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTINOMIAL_CLASSIFICATION = "multinomial_classification"
    CLUSTERING = "clustering"
    RECOMMENDER = "recommender"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction/representation"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    TOPIC_MODELING = "topic_modeling"
    NER = "ner"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_LOCALIZATION = "object_localization"
    OTHER = "other"


class Framework(str, metaclass=ExtendedEnumMeta):
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    MXNET = "mxnet"
    KERAS = "keras"
    LIGHT_GBM = "lightgbm"
    PYMC3 = "pymc3"
    PYOD = "pyod"
    SPACY = "spacy"
    PROPHET = "prophet"
    SKTIME = "sktime"
    STATSMODELS = "statsmodels"
    CUML = "cuml"
    ORACLE_AUTOML = "oracle_automl"
    H20 = "h2o"
    TRANSFORMERS = "transformers"
    NLTK = "nltk"
    EMCEE = "emcee"
    PYSTAN = "pystan"
    BERT = "bert"
    GENSIM = "gensim"
    FLAIR = "flair"
    WORD2VEC = "word2vec"
    ENSEMBLE = "ensemble"
    SPARK = "pyspark"
    OTHER = "other"


class ModelMetadataItem(ABC):
    """The base abstract class representing model metadata item.

    Methods
    -------
    to_dict(self) -> Dict
        Serializes model metadata item to dictionary.
    from_dict(cls, data: Dict) -> ModelMetadataItem
        Constructs an instance of ModelMetadataItem from a dictionary.
    to_yaml(self)
        Serializes model metadata item to YAML.
    size(self) -> int
        Returns the size of the metadata in bytes.
    to_json(self) -> JSON
        Serializes metadata item to JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata item value to a local file or object storage.
    validate(self) -> bool
        Validates metadata item.
    """

    _FIELDS = []

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadataItem":
        """Constructs an instance of `ModelMetadataItem` from a dictionary.

        Parameters
        ----------
        data : Dict
            Metadata item in a dictionary format.

        Returns
        -------
        ModelMetadataItem
            An instance of model metadata item.
        """
        return cls(**data or {})

    def to_dict(self) -> dict:
        """Serializes model metadata item to dictionary.

        Returns
        -------
        dict
            The dictionary representation of model metadata item.
        """
        return {field: getattr(self, field) for field in self._FIELDS}

    def to_yaml(self):
        """Serializes model metadata item to YAML.

        Returns
        -------
        Yaml
            The model metadata item in a YAML representation.
        """
        return yaml.dump(self.to_dict(), Dumper=dumper)

    def size(self) -> int:
        """Returns the size of the model metadata in bytes.

        Returns
        -------
        int
            The size of model metadata in bytes.
        """
        return len(json.dumps(self.to_dict()).encode("utf-16"))

    def to_json(self):
        """Serializes metadata item into a JSON.

        Returns
        -------
        JSON
            The metadata item in a JSON representation.
        """
        return json.dumps(self.to_dict())

    def to_json_file(
        self,
        file_path: str,
        storage_options: dict = None,
    ) -> None:
        """Saves the metadata item value to a local file or object storage.

        Parameters
        ----------
        file_path : str
            The file path to store the data.
            "oci://bucket_name@namespace/folder_name/"
            "oci://bucket_name@namespace/folder_name/result.json"
            "path/to/local/folder"
            "path/to/local/folder/result.json"
        storage_options : dict. Default None
            Parameters passed on to the backend filesystem class.
            Defaults to `options` set using `DatasetFactory.set_default_storage()`.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
            ValueError: When file path is empty.
            TypeError: When file path not a string.

        Examples
        --------
        >>> metadata_item = ModelCustomMetadataItem(key="key1", value="value1")
        >>> storage_options = {"config": oci.config.from_file(os.path.join("~/.oci", "config"))}
        >>> storage_options
        {'log_requests': False,
            'additional_user_agent': '',
            'pass_phrase': None,
            'user': '<user-id>',
            'fingerprint': '05:15:2b:b1:46:8a:32:ec:e2:69:5b:32:01:**:**:**)',
            'tenancy': '<tenency-id>',
            'region': 'us-ashburn-1',
            'key_file': '/home/datascience/.oci/oci_api_key.pem'}
        >>> metadata_item.to_json_file(file_path = 'oci://bucket_name@namespace/folder_name/file.json', storage_options=storage_options)
        >>> metadata_item.to_json_file("path/to/local/folder/file.json")
        """
        if not file_path:
            raise ValueError("File path must be specified.")

        if not isinstance(file_path, str):
            raise TypeError("File path must be a string.")

        if not Path(os.path.basename(file_path)).suffix:
            file_path = os.path.join(file_path, f"{self.key}.json")

        if not storage_options:
            storage_options = factory.default_storage_options or {"config": {}}

        with fsspec.open(
            file_path,
            mode="w",
            **(storage_options),
        ) as f:
            f.write(json.dumps(self.value))

    def _to_oci_metadata(self):
        """Converts metadata item to OCI metadata item."""
        dict = self.to_dict()
        if not dict["value"]:
            return oci.data_science.models.Metadata(**dict)
        if isinstance(dict["value"], (str, int, float)):
            dict["value"] = str(dict["value"]).replace("NaN", "null")
        else:
            dict["value"] = json.dumps(dict["value"]).replace("NaN", "null")
        return oci.data_science.models.Metadata(**dict)

    @classmethod
    def _from_oci_metadata(cls, oci_metadata_item) -> "ModelMetadataItem":
        """Creates a new metadata item from the OCI metadata item."""
        oci_metadata_item = to_dict(oci_metadata_item)
        key_value_map = {field: oci_metadata_item.get(field) for field in cls._FIELDS}

        if isinstance(key_value_map["value"], str):
            try:
                key_value_map["value"] = json.loads(oci_metadata_item.get("value"))
            except Exception:
                pass

        return cls(**key_value_map)

    def __hash__(self):
        return hash(self.key.lower())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return self.to_yaml()

    @abstractmethod
    def validate(self) -> bool:
        """Validates metadata item.

        Returns
        -------
        bool
            True if validation passed.
        """
        pass


class ModelTaxonomyMetadataItem(ModelMetadataItem):
    """Class that represents model taxonomy metadata item.

    Attributes
    ----------
    key: str
        The model metadata item key.
    value: str
        The model metadata item value.

    Methods
    -------
    reset(self) -> None
        Resets model metadata item.
    to_dict(self) -> Dict
        Serializes model metadata item to dictionary.
    from_dict(cls) -> ModelTaxonomyMetadataItem
        Constructs model metadata item from dictionary.
    to_yaml(self)
        Serializes model metadata item to YAML.
    size(self) -> int
        Returns the size of the metadata in bytes.
    update(self, value: str = "") -> None
        Updates metadata item information.
    to_json(self) -> JSON
        Serializes metadata item into a JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata item value to a local file or object storage.
    validate(self) -> bool
        Validates metadata item.
    """

    _FIELDS = ["key", "value"]

    def __init__(
        self,
        key: str,
        value: str = None,
    ):
        self.key = key
        self.value = value

    @property
    def key(self) -> str:
        return self._key

    @key.setter
    def key(self, key: str):
        """The model metadata key setter.

        Raises
        ------
        TypeError
            If provided key is not a string.
        ValueError
            If provided key is already setup.
            If provided key is empty.
        """
        if hasattr(self, "_key"):
            raise ValueError("The key field is immutable and cannot be changed.")
        if not isinstance(key, str):
            raise TypeError("The key must be a string.")
        if key is None or key == "":
            raise ValueError("The key cannot be empty.")
        self._key = key

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str):
        """The model metadata value setter. Accepts any JSON serializable value.

        Raises
        ------
        ValueError
            If provided value cannot be serialized to JSON.
        """
        if value is None or value == "":
            self._value = value
            return

        try:
            json.dumps(value)
        except TypeError:
            raise ValueError(
                f"An error occurred in attempt to serialize the value of {self.key} to JSON. "
                "The value must be JSON serializable."
            )

        self._value = value

    def reset(self) -> None:
        """Resets model metadata item.

        Resets value to None.

        Returns
        -------
        None
            Nothing.
        """
        self.update(value=None)

    def update(self, value: str) -> None:
        """Updates metadata item value.

        Parameters
        ----------
        value: str
            The value of model metadata item.

        Returns
        -------
        None
            Nothing.
        """
        self.value = value

    def validate(self) -> bool:
        """Validates metadata item.

        Returns
        -------
        bool
            True if validation passed.

        Raises
        ------
        ValueError
            If invalid UseCaseType provided.
            If invalid Framework provided.
        """
        if (
            self.key.lower() == MetadataTaxonomyKeys.USE_CASE_TYPE.lower()
            and self.value
            and (not isinstance(self.value, str) or self.value not in UseCaseType)
        ):
            raise ValueError(
                f"Invalid value of `UseCaseType`. Choose from {UseCaseType.values()}."
            )
        if (
            self.key.lower() == MetadataTaxonomyKeys.FRAMEWORK.lower()
            and self.value
            and (not isinstance(self.value, str) or self.value not in Framework)
        ):
            raise ValueError(
                f"Invalid value of `Framework`. Choose from {Framework.values()}."
            )
        return True


class ModelCustomMetadataItem(ModelTaxonomyMetadataItem):
    """Class that represents model custom metadata item.

    Attributes
    ----------
    key: str
        The model metadata item key.
    value: str
        The model metadata item value.
    description: str
        The model metadata item description.
    category: str
        The model metadata item category.

    Methods
    -------
    reset(self) -> None
        Resets model metadata item.
    to_dict(self)->dict
        Serializes model metadata item to dictionary.
    from_dict(cls) -> ModelCustomMetadataItem
        Constructs model metadata item from dictionary.
    to_yaml(self)
        Serializes model metadata item to YAML.
    size(self) -> int
        Returns the size of the metadata in bytes.
    update(self, value: str = "", description: str = "", category: str = "") -> None
        Updates metadata item information.
    to_json(self) -> JSON
        Serializes metadata item into a JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata item value to a local file or object storage.
    validate(self) -> bool
        Validates metadata item.
    """

    _FIELDS = ["key", "value", "description", "category"]

    def __init__(
        self,
        key: str,
        value: str = None,
        description: str = None,
        category: str = None,
    ):
        super().__init__(key=key, value=value)
        self.description = description
        self.category = category

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, description: str):
        """The model metadata description setter.

        Raises
        ------
        TypeError
            If provided key is not a string.
        """
        if description != None and not isinstance(description, str):
            raise TypeError("The description must be a string.")

        self._description = description

    @property
    def category(self) -> str:
        return self._category

    @category.setter
    def category(self, category: str):
        """The model metadata category setter.

        Raises
        ------
        TypeError
            If provided category is not a string.
        ValueError
            If provided category not supported.
        """
        if not category:
            self._category = None
            return

        if not isinstance(category, str):
            raise TypeError(
                f"Invalid category type for the {self.key}."
                "The category must be a string."
            )

        if category not in MetadataCustomCategory:
            raise ValueError(
                f"Invalid category value for the {self.key}. "
                f"Choose from {MetadataCustomCategory.values()}."
            )

        self._category = category

    def reset(self) -> None:
        """Resets model metadata item.

        Resets value, description and category to None.

        Returns
        -------
        None
            Nothing.
        """
        self.update(value=None, description=None, category=None)

    def update(self, value: str, description: str, category: str) -> None:
        """Updates metadata item.

        Parameters
        ----------
        value: str
            The value of model metadata item.
        description: str
            The description of model metadata item.
        category: str
            The category of model metadata item.

        Returns
        -------
        None
            Nothing.
        """
        self.value = value
        self.description = description
        self.category = category

    def _to_oci_metadata(self):
        """Converts metadata item to OCI metadata item."""
        oci_metadata_item = super()._to_oci_metadata()
        if not oci_metadata_item.value:
            oci_metadata_item.value = _METADATA_EMPTY_VALUE
        if not oci_metadata_item.category:
            oci_metadata_item.category = MetadataCustomCategory.OTHER
        return oci_metadata_item

    def validate(self) -> bool:
        """Validates metadata item.

        Returns
        -------
        bool
            True if validation passed.

        Raises
        ------
        ValueError
            If invalid category provided.
        MetadataValueTooLong
            If value exceeds the length limit.
        """
        if self.category and self.category not in MetadataCustomCategory:
            raise ValueError(
                f"Invalid category value for the {self.key}. "
                f"Choose from {MetadataCustomCategory.values()}."
            )

        if self.value:
            value = (
                self.value if isinstance(self.value, str) else json.dumps(self.value)
            )
            if len(value) > METADATA_VALUE_LENGTH_LIMIT:
                raise MetadataValueTooLong(self.key, len(value))

        if (
            self.description
            and len(self.description) > METADATA_DESCRIPTION_LENGTH_LIMIT
        ):
            raise MetadataDescriptionTooLong(self.key, len(self.description))

        return True


class ModelMetadata(ABC):
    """The base abstract class representing model metadata.

    Methods
    -------
    get(self, key: str) -> ModelMetadataItem
        Returns the model metadata item by provided key.
    reset(self) -> None
        Resets all model metadata items to empty values.
    to_dataframe(self) -> pd.DataFrame
        Returns the model metadata list in a data frame format.
    size(self) -> int
        Returns the size of the model metadata in bytes.
    validate(self) -> bool
        Validates metadata.
    to_dict(self)
        Serializes model metadata into a dictionary.
    from_dict(cls) -> ModelMetadata
        Constructs model metadata from dictionary.
    to_yaml(self)
        Serializes model metadata into a YAML.
    to_json(self)
        Serializes model metadata into a JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata to a local file or object storage.
    """

    @abstractmethod
    def __init__(self):
        """Initializes Model Metadata."""
        self._items = set()

    def get(self, key: str) -> ModelMetadataItem:
        """Returns the model metadata item by provided key.

        Parameters
        ----------
        key: str
            The key of model metadata item.

        Returns
        -------
        ModelMetadataItem
            The model metadata item.

        Raises
        ------
        ValueError
            If provided key is empty or metadata item not found.
        """
        if key is None or not isinstance(key, str) or key == "":
            raise ValueError("The key must not be an empty string.")
        for item in self._items:
            if item.key.lower() == key.lower():
                return item
        raise ValueError(f"The metadata with {key} not found.")

    def reset(self) -> None:
        """Resets all model metadata items to empty values.

        Resets value, description and category to None for every metadata item.
        """
        for item in self._items:
            item.reset()

    def size(self) -> int:
        """Returns the size of the model metadata in bytes.

        Returns
        -------
        int
            The size of model metadata in bytes.
        """
        return sum(item.size() for item in self._items)

    def validate_size(self) -> bool:
        """Validates model metadata size.

        Validates the size of metadata. Throws an error if the size of the metadata
        exceeds expected value.

        Returns
        -------
        bool
            True if metadata size is valid.

        Raises
        ------
        MetadataSizeTooLarge
            If the size of the metadata exceeds expected value.
        """
        if self.size() > METADATA_SIZE_LIMIT:
            raise MetadataSizeTooLarge(self.size())
        return True

    def validate(self) -> bool:
        """Validates model metadata.

        Returns
        -------
        bool
            True if metadata is valid.
        """
        for item in self._items:
            item.validate()
        return True

    def to_dict(self):
        """Serializes model metadata into a dictionary.

        Returns
        -------
        Dict
            The model metadata in a dictionary representation.
        """
        return {"data": [item.to_dict() for item in self._items]}

    def to_yaml(self):
        """Serializes model metadata into a YAML.

        Returns
        -------
        Yaml
            The model metadata in a YAML representation.
        """
        return yaml.dump(self.to_dict(), Dumper=dumper)

    def to_json(self):
        """Serializes model metadata into a JSON.

        Returns
        -------
        JSON
            The model metadata in a JSON representation.
        """
        return json.dumps(self.to_dict())

    @property
    def keys(self) -> Tuple[str]:
        """Returns all registered metadata keys.

        Returns
        -------
        Tuple[str]
            The list of metadata keys.
        """
        return tuple(item.key for item in self._items)

    def _to_oci_metadata(self):
        """Convert to a list of `oci.data_science.models.Metadata` objects.

        Returns
        -------
        list[oci.data_science.models.Metadata]
            A list of oci data science model metadata.

        Examples
        --------
        >>> metadata_taxonomy = ModelTaxonomyMetadata()
        >>> metadata_taxonomy.get(key="FrameworkVersion").update(value="2.3.1")
        >>> metadata_taxonomy._to_oci_metadata()
        [{
        "key": "FrameworkVersion",
        "value": "2.3.1"
        },
        {
        "key": "UseCaseType",
        "value": null
        },
        {
        "key": "Algorithm",
        "value": null
        },
        {
        "key": "Framework",
        "value": null
        },
        {
        "key": "Hyperparameters",
        "value": null
        }]
        """
        return [item._to_oci_metadata() for item in self._items]

    def to_json_file(
        self,
        file_path: str,
        storage_options: dict = None,
    ) -> None:
        """Saves the metadata to a local file or object storage.

        Parameters
        ----------
        file_path : str
            The file path to store the data.
            "oci://bucket_name@namespace/folder_name/"
            "oci://bucket_name@namespace/folder_name/metadata.json"
            "path/to/local/folder"
            "path/to/local/folder/metadata.json"
        storage_options : dict. Default None
            Parameters passed on to the backend filesystem class.
            Defaults to `options` set using `DatasetFactory.set_default_storage()`.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
            ValueError: When file path is empty.
            TypeError: When file path not a string.

        Examples
        --------
        >>> metadata = ModelTaxonomyMetadataItem()
        >>> storage_options = {"config": oci.config.from_file(os.path.join("~/.oci", "config"))}
        >>> storage_options
        {'log_requests': False,
            'additional_user_agent': '',
            'pass_phrase': None,
            'user': '<user-id>',
            'fingerprint': '05:15:2b:b1:46:8a:32:ec:e2:69:5b:32:01:**:**:**)',
            'tenancy': '<tenancy-id>',
            'region': 'us-ashburn-1',
            'key_file': '/home/datascience/.oci/oci_api_key.pem'}
        >>> metadata.to_json_file(file_path = 'oci://bucket_name@namespace/folder_name/metadata_taxonomy.json', storage_options=storage_options)
        >>> metadata_item.to_json_file("path/to/local/folder/metadata_taxonomy.json")
        """
        if not file_path:
            raise ValueError("File path must be specified.")

        if not isinstance(file_path, str):
            raise TypeError("File path must be a string.")

        if not Path(os.path.basename(file_path)).suffix:
            file_path = os.path.join(file_path, f"{self.__class__.__name__}.json")

        if not storage_options:
            storage_options = factory.default_storage_options or {"config": {}}

        with fsspec.open(
            file_path,
            mode="w",
            **(storage_options),
        ) as f:
            f.write(self.to_json())

    def __getitem__(self, key: str) -> ModelMetadataItem:
        return self.get(key)

    def __repr__(self):
        return self.to_yaml()

    def __len__(self):
        return len(self._items)

    @abstractclassmethod
    def _from_oci_metadata(cls, metadata_list):
        pass

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Returns the model metadata list in a data frame format.

        Returns
        -------
        `pandas.DataFrame`
            The model metadata in a dataframe format.
        """
        pass

    @abstractclassmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        """Constructs an instance of `ModelMetadata` from a dictionary.

        Parameters
        ----------
        data : Dict
            Model metadata in a dictionary format.

        Returns
        -------
        ModelMetadata
            An instance of model metadata.
        """
        pass


class ModelCustomMetadata(ModelMetadata):
    """Class that represents Model Custom Metadata.

    Methods
    -------
    get(self, key: str) -> ModelCustomMetadataItem
        Returns the model metadata item by provided key.
    reset(self) -> None
        Resets all model metadata items to empty values.
    to_dataframe(self) -> pd.DataFrame
        Returns the model metadata list in a data frame format.
    size(self) -> int
        Returns the size of the model metadata in bytes.
    validate(self) -> bool
        Validates metadata.
    to_dict(self)
        Serializes model metadata into a dictionary.
    from_dict(cls) -> ModelCustomMetadata
        Constructs model metadata from dictionary.
    to_yaml(self)
        Serializes model metadata into a YAML.
    add(self, key: str, value: str, description: str = "", category: str = MetadataCustomCategory.OTHER, replace: bool = False) -> None:
        Adds a new model metadata item. Replaces existing one if replace flag is True.
    remove(self, key: str) -> None
        Removes a model metadata item by key.
    clear(self) -> None
        Removes all metadata items.
    isempty(self) -> bool
        Checks if metadata is empty.
    to_json(self)
        Serializes model metadata into a JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata to a local file or object storage.

    Examples
    --------
    >>> metadata_custom = ModelCustomMetadata()
    >>> metadata_custom.add(key="format", value="pickle")
    >>> metadata_custom.add(key="note", value="important note", description="some description")
    >>> metadata_custom["format"].description = "some description"
    >>> metadata_custom.to_dataframe()
                        Key              Value         Description      Category
    ----------------------------------------------------------------------------
    0                format             pickle    some description  user defined
    1                  note     important note    some description  user defined
    >>> metadata_custom
        metadata:
        - category: user defined
          description: some description
          key: format
          value: pickle
        - category: user defined
          description: some description
          key: note
          value: important note
    >>> metadata_custom.remove("format")
    >>> metadata_custom
        metadata:
        - category: user defined
          description: some description
          key: note
          value: important note
    >>> metadata_custom.to_dict()
        {'metadata': [{
                'key': 'note',
                'value': 'important note',
                'category': 'user defined',
                'description': 'some description'
            }]}
    >>> metadata_custom.reset()
    >>> metadata_custom
        metadata:
        - category: None
          description: None
          key: note
          value: None
    >>> metadata_custom.clear()
    >>> metadata_custom.to_dataframe()
                        Key              Value         Description      Category
    ----------------------------------------------------------------------------
    """

    def __init__(self):
        """Initializes custom model metadata."""
        self._items = set()

    def add(
        self,
        key: str,
        value: str,
        description: str = "",
        category: str = MetadataCustomCategory.OTHER,
        replace: bool = False,
    ) -> None:
        """Adds a new model metadata item. Overrides the existing one if replace flag is True.

        Parameters
        ----------
        key: str
            The metadata item key.
        value: str
            The metadata item value.
        description: str
            The metadata item description.
        category: str
            The metadata item category.
        replace: bool
            Overrides the existing metadata item if replace flag is True.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            If provided key is not a string.
            If provided description not a string.
        ValueError
            If provided key is empty.
            If provided value is empty.
            If provided value cannot be serialized to JSON.
            If item with provided key is already registered and replace flag is False.
            If provided category is not supported.
        MetadataValueTooLong
            If the length of provided value exceeds 255 charracters.
        MetadataDescriptionTooLong
            If the length of provided description exceeds 255 charracters.
        """

        if not category:
            category = MetadataCustomCategory.OTHER

        if key is None or key == "":
            raise ValueError("The key cannot be empty.")

        if value is None or value == "":
            raise ValueError("The value cannot be empty.")

        if not isinstance(key, str):
            raise TypeError("The key must be a string.")

        if not isinstance(category, str):
            raise TypeError("The category must be a string.")

        if category not in MetadataCustomCategory:
            raise ValueError(
                f"Invalid category value. "
                f"Choose from {MetadataCustomCategory.values()}."
            )

        if description and not isinstance(description, str):
            raise TypeError("The description must be a string.")

        try:
            tmp_value = json.dumps(value)
        except TypeError:
            raise ValueError(
                f"An error occurred in attempt to serialize the value of `{key}` to JSON. "
                "The value must be JSON serializable."
            )

        if len(tmp_value) > METADATA_VALUE_LENGTH_LIMIT:
            raise MetadataValueTooLong(key, len(tmp_value))

        if description and len(description) > METADATA_DESCRIPTION_LENGTH_LIMIT:
            raise MetadataDescriptionTooLong(key, len(description))

        if not replace and key in self.keys:
            raise ValueError(
                f"The metadata item with key {key} is already registered. "
                "Use replace=True to overwrite."
            )

        self._add(
            ModelCustomMetadataItem(
                key=key,
                value=value,
                description=description,
                category=category,
            ),
            replace=replace,
        )

    def _add(self, item: ModelCustomMetadataItem, replace=False) -> None:
        """Adds a new model metadata item.

        Overrides the existing one if replace flag is True.

        Parameters
        ----------
        item: ModelCustomMetadataItem
            The model metadata item.
        replace: bool
            Overrides the existing metadata item if replace flag is True.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If item is already registered and replace flag is False.
        TypeError
            If input data has a wrong format.
        """
        if not isinstance(item, ModelCustomMetadataItem):
            raise TypeError(
                "Argument must be an instance of the class ModelCustomMetadataItem."
            )
        if not replace and item in self._items:
            raise ValueError(
                f"The metadata item with key {item.key} is already registered. "
                "Use replace=True to overwrite."
            )
        self._items.discard(item)
        self._items.add(item)

    def _add_many(self, items: List[ModelCustomMetadataItem], replace=False) -> None:
        """Adds model metadata items into model metadata.

        Overrides the existing ones if replace flag is True.

        Parameters
        ----------
        items: List[ModelCustomMetadataItem]
            The list of model metadata items.
        replace: bool
            Overrides the existing metadata items if replace flag is True.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            If input data has wrong format.
        """
        if not isinstance(items, list):
            raise TypeError("Argument must be a list of model metadata items.")
        for item in items:
            self._add(item, replace)

    def set_training_data(self, path: str, data_size: str = None):
        """Adds training_data path and data size information into model custom metadata.

        Parameters
        ----------
        path: str
            The path where the training_data is stored.
        data_size: str
            The size of the training_data.

        Returns
        -------
        None
            Nothing.
        """
        self.add(
            key=MetadataCustomKeys.TRAINING_DATASET,
            value=path,
            category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
            description="The path where training dataset path are stored.",
            replace=True,
        )
        if data_size is not None:
            self.add(
                key=MetadataCustomKeys.TRAINING_DATASET_SIZE,
                value=data_size,
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
                description="The size of the training data.",
                replace=True,
            )

    def set_validation_data(self, path: str, data_size: str = None):
        """Adds validation_data path and data size information into model custom metadata.

        Parameters
        ----------
        path: str
            The path where the validation_data is stored.
        data_size: str
            The size of the validation_data.

        Returns
        -------
        None
            Nothing.
        """
        self.add(
            key=MetadataCustomKeys.VALIDATION_DATASET,
            value=path,
            category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
            description="The path where validation dataset path are stored.",
            replace=True,
        )
        if data_size is not None:
            self.add(
                key=MetadataCustomKeys.VALIDATION_DATASET_SIZE,
                value=data_size,
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
                description="The size of the validation data.",
                replace=True,
            )

    def remove(self, key: str) -> None:
        """Removes a model metadata item.

        Parameters
        ----------
        key: str
            The key of the metadata item that should be removed.

        Returns
        -------
        None
            Nothing.
        """
        self._items.discard(self.get(key))

    def clear(self) -> None:
        """Removes all metadata items.

        Returns
        -------
        None
            Nothing.
        """
        self._items.clear()

    def isempty(self) -> bool:
        """Checks if metadata is empty.

        Returns
        -------
        bool
            True if metadata is empty, False otherwise.
        """
        return len(self._items) == 0

    @classmethod
    def _from_oci_metadata(cls, metadata_list):
        """Convert from list of OCI metadata list to an ModelCustomMetadata object.

        Returns
        -------
        ModelCustomMetadata
            A `ModelCustomMetadata` instance.
        """
        metadata = cls()
        for item in metadata_list:
            metadata._add(
                ModelCustomMetadataItem._from_oci_metadata(item), replace=True
            )
        return metadata

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the model metadata list in a data frame format.

        Returns
        -------
        `pandas.DataFrame`
            The model metadata in a dataframe format.
        """
        return (
            pd.DataFrame(
                (
                    (item.key, item.value, item.description, item.category)
                    for item in self._items
                ),
                columns=[value for value in MetadataCustomPrintColumns.values()],
            )
            .sort_values(by=[MetadataCustomPrintColumns.KEY])
            .reset_index(drop=True)
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelCustomMetadata":
        """Constructs an instance of `ModelCustomMetadata` from a dictionary.

        Parameters
        ----------
        data : Dict
            Model metadata in a dictionary format.

        Returns
        -------
        ModelCustomMetadata
            An instance of model custom metadata.

        Raises
        ------
        ValueError
            In case of the wrong input data format.
        """
        if (
            not data
            or not isinstance(data, Dict)
            or not "data" in data
            or not isinstance(data["data"], List)
        ):
            raise ValueError(
                "An error occurred when attempting to deserialize the model custom metadata from a dictionary. "
                "The input data must be a dictionary with `data` key. Example: `{'data': []}`"
            )

        metadata = cls()
        for item in data["data"]:
            metadata._add(ModelCustomMetadataItem.from_dict(item), replace=True)
        return metadata


class ModelTaxonomyMetadata(ModelMetadata):
    """Class that represents Model Taxonomy Metadata.

    Methods
    -------
    get(self, key: str) -> ModelTaxonomyMetadataItem
        Returns the model metadata item by provided key.
    reset(self) -> None
        Resets all model metadata items to empty values.
    to_dataframe(self) -> pd.DataFrame
        Returns the model metadata list in a data frame format.
    size(self) -> int
        Returns the size of the model metadata in bytes.
    validate(self) -> bool
        Validates metadata.
    to_dict(self)
        Serializes model metadata into a dictionary.
    from_dict(cls) -> ModelTaxonomyMetadata
        Constructs model metadata from dictionary.
    to_yaml(self)
        Serializes model metadata into a YAML.
    to_json(self)
        Serializes model metadata into a JSON.
    to_json_file(self, file_path: str, storage_options: dict = None) -> None
        Saves the metadata to a local file or object storage.

    Examples
    --------
    >>> metadata_taxonomy = ModelTaxonomyMetadata()
    >>> metadata_taxonomy.to_dataframe()
                    Key                   Value
    --------------------------------------------
    0        UseCaseType   binary_classification
    1          Framework                 sklearn
    2   FrameworkVersion                   0.2.2
    3          Algorithm               algorithm
    4    Hyperparameters                      {}

    >>> metadata_taxonomy.reset()
    >>> metadata_taxonomy.to_dataframe()
                    Key                    Value
    --------------------------------------------
    0        UseCaseType                    None
    1          Framework                    None
    2   FrameworkVersion                    None
    3          Algorithm                    None
    4    Hyperparameters                    None

    >>> metadata_taxonomy
        metadata:
        - key: UseCaseType
          category: None
          description: None
          value: None
    """

    def __init__(self):
        super().__init__()
        for key in MetadataTaxonomyKeys.values():
            self._items.add(ModelTaxonomyMetadataItem(key=key))

    def _populate_from_map(self, map: Dict[str, str]) -> None:
        """Populates metadata information from map.

        Parameters
        ----------
        map: Dict[str, str]
            The key/value map with model metadata information.

        Returns
        -------
        None
            Nothing.
        """
        for value in MetadataTaxonomyKeys.values():
            if value in map:
                self[value].update(value=map[value])

    @classmethod
    def _from_oci_metadata(cls, metadata_list):
        """
        Convert from list of oci metadata to a ModelTaxonomyMetadata object.

        Parameters
        ----------
        metadata_list: List
            List of oci metadata.

        Returns
        -------
        ModelTaxonomyMetadata
            A `ModelTaxonomyMetadata` instance.
        """
        metadata = cls()
        for oci_item in metadata_list:
            item = ModelTaxonomyMetadataItem._from_oci_metadata(oci_item)
            metadata[item.key].update(value=item.value)
        return metadata

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the model metadata list in a data frame format.

        Returns
        -------
        `pandas.DataFrame`
            The model metadata in a dataframe format.
        """
        return (
            pd.DataFrame(
                ((item.key, item.value) for item in self._items),
                columns=[value for value in MetadataTaxonomyPrintColumns.values()],
            )
            .sort_values(by=[MetadataTaxonomyPrintColumns.KEY])
            .reset_index(drop=True)
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelTaxonomyMetadata":
        """Constructs an instance of `ModelTaxonomyMetadata` from a dictionary.

        Parameters
        ----------
        data : Dict
            Model metadata in a dictionary format.

        Returns
        -------
        ModelTaxonomyMetadata
            An instance of model taxonomy metadata.

        Raises
        ------
        ValueError
            In case of the wrong input data format.
        """
        if (
            not data
            or not isinstance(data, Dict)
            or not "data" in data
            or not isinstance(data["data"], List)
        ):
            raise ValueError(
                "An error occurred when attempting to deserialize the model taxonomy metadata from a dictionary. "
                "The input data must be a dictionary with `data` key. Example: `{'data': []}`"
            )

        metadata = cls()
        for item in data["data"]:
            item = ModelTaxonomyMetadataItem.from_dict(item)
            metadata[item.key].update(value=item.value)
        return metadata


@dataclass(repr=True)
class ModelProvenanceMetadata(DataClassSerializable):
    """ModelProvenanceMetadata class.

    Examples
    --------
    >>> provenance_metadata = ModelProvenanceMetadata.fetch_training_code_details()
    ModelProvenanceMetadata(repo=<git.repo.base.Repo '/home/datascience/.git'>, git_branch='master', git_commit='99ad04c31803f1d4ffcc3bf4afbd6bcf69a06af2', repository_url='file:///home/datascience', "", "")
    >>> provenance_metadata.assert_path_not_dirty("your_path", ignore=False)
    """

    repo: str = field(default=None, metadata={"serializable": False})
    git_branch: str = None
    git_commit: str = None
    repository_url: str = None
    training_script_path: str = None
    training_id: str = None
    artifact_dir: str = None

    @classmethod
    def fetch_training_code_details(
        cls,
        training_script_path: str = None,
        training_id: str = None,
        artifact_dir: str = None,
    ):
        """Fetches the training code details: repo, git_branch, git_commit, repository_url, training_script_path and training_id.

        Parameters
        ----------
        training_script_path: (str, optional). Defaults to None.
            Training script path.
        training_id: (str, optional). Defaults to None.
            The training OCID for model.
        artifact_dir: str
            artifact directory to store the files needed for deployment.

        Returns
        -------
        ModelProvenanceMetadata
            A ModelProvenanceMetadata instance.
        """
        git_dir = CURRENT_WORKING_DIR
        if training_script_path:
            if not os.path.exists(training_script_path):
                logger.warning(
                    f"Training script {os.path.abspath(training_script_path)} does not exist."
                )
            else:
                training_script_path = os.path.abspath(training_script_path)
                git_dir = os.path.dirname(training_script_path)
        repo = git.Repo(git_dir, search_parent_directories=True)
        # get repository url
        if len(repo.remotes) > 0:
            repository_url = (
                repo.remotes.origin.url
                if repo.remotes.origin in repo.remotes
                else list(repo.remotes.values())[0].url
            )
        else:
            repository_url = "file://" + repo.working_dir  # no remote repo

        # get git commit
        git_branch = None
        git_commit = None

        try:
            # get git branch
            git_branch = format(repo.active_branch)
            git_commit = format(str(repo.head.commit.hexsha)) or None
        except Exception:
            logger.warning("No commit found.")
        return cls(
            repo=repo,
            git_branch=git_branch,
            git_commit=git_commit,
            repository_url=repository_url,
            training_script_path=training_script_path,
            training_id=training_id,
            artifact_dir=artifact_dir,
        )

    def assert_path_not_dirty(self, path: str, ignore: bool):
        """Checks if all the changes in this path has been commited.

        Parameters
        ----------
        path: (str)
            path.
        ignore (bool)
            whether to ignore the changes or not.

        Raises
        ------
        ChangesNotCommitted: if there are changes not being commited.

        Returns
        -------
        None
            Nothing.
        """
        if ObjectStorageDetails.is_oci_path(path):
            return

        if self.repo is not None and not ignore:
            path_abs = os.path.abspath(path)
            if (
                os.path.commonpath([path_abs, self.repo.working_dir])
                == self.repo.working_dir
            ):
                path_relpath = os.path.relpath(path_abs, self.repo.working_dir)
                if self.repo.is_dirty(path=path_relpath) or any(
                    [
                        os.path.commonpath([path_relpath, untracked]) == path_relpath
                        for untracked in self.repo.untracked_files
                    ]
                ):
                    raise ChangesNotCommitted(path_abs)

    def _to_oci_metadata(self) -> oci.data_science.models.ModelProvenance:
        """Convert to `oci.data_science.models.ModelProvenance` object.

        Returns
        -------
        oci.data_science.models.ModelProvenance
            OCI model provenance object.
        """
        return oci.data_science.models.ModelProvenance(
            repository_url=self.repository_url,
            git_branch=self.git_branch,
            git_commit=self.git_commit,
            script_dir=self.artifact_dir,
            training_script=self.training_script_path,
            training_id=self.training_id,
        )

    @classmethod
    def _from_oci_metadata(
        cls, model_provenance: oci.data_science.models.ModelProvenance
    ) -> "ModelProvenanceMetadata":
        """Creates a new model provenance metadata item from the `oci.data_science.models.ModelProvenance` object.

        Returns
        -------
        ModelProvenanceMetadata
            Model provenance metadata object.
        """
        return ModelProvenanceMetadata(
            repo=model_provenance.repository_url,
            git_branch=model_provenance.git_branch,
            git_commit=model_provenance.git_commit,
            repository_url=model_provenance.repository_url,
            training_script_path=model_provenance.training_script,
            training_id=model_provenance.training_id,
            artifact_dir=model_provenance.script_dir,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ModelProvenanceMetadata":
        """Constructs an instance of ModelProvenanceMetadata from a dictionary.

        Parameters
        ----------
        data : Dict[str,str]
            Model provenance metadata in dictionary format.

        Returns
        -------
        ModelProvenanceMetadata
            An instance of ModelProvenanceMetadata.
        """
        return cls(**data or {})

    def to_dict(self) -> dict:
        """Serializes model provenance metadata into a dictionary.

        Returns
        -------
        Dict
            The dictionary representation of the model provenance metadata.
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if ("serializable" not in f.metadata or f.metadata["serializable"])
        }

    def __repr__(self):
        """Returns printable version of object.

        Parameters
        ----------
        string
            Serialized version of object as a YAML string
        """
        return self.to_yaml()
