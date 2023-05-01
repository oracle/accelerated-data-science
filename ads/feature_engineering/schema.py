#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import asteval
import fsspec
import json
import os
import sys
import yaml
from abc import ABC, abstractmethod
from cerberus import Validator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from string import Template
from os import path
from ads.common.serializer import DataClassSerializable
from ads.common.object_storage_details import ObjectStorageDetails

try:
    from yaml import CDumper as dumper
    from yaml import CLoader as loader
except:
    from yaml import Dumper as dumper
    from yaml import Loader as loader

SCHEMA_VALIDATOR_NAME = "data_schema.json"
INPUT_OUTPUT_SCHENA_SIZE_LIMIT = 32000
SCHEMA_VERSION = "1.1"
DEFAULT_SCHEMA_VERSION = "1.0"
SCHEMA_KEY = "schema"
SCHEMA_VERSION_KEY = "version"
DEFAULT_STORAGE_OPTIONS = None


class SchemaSizeTooLarge(ValueError):
    def __init__(self, size: int):
        super().__init__(
            f"The schema `{size}` bytes and "
            f"the maximum allowable schema size is `{INPUT_OUTPUT_SCHENA_SIZE_LIMIT}` bytes. "
            "Reduce the size of the schema."
        )


class DataSizeTooWide(ValueError):
    def __init__(self, data_col_num: int, max_col_num: int):
        super().__init__(
            f"The data has `{data_col_num}` columns and "
            f"the maximum allowable number of columns is `{max_col_num}`. "
            "Increase allowable number of columns by setting a larger max_col_num. It will take longer time to prepare."
        )


@dataclass(repr=False)
class Expression(DataClassSerializable):
    """
    Expression allows specifying string representation of an expression which can be evaluated by the language corresponding to the value provided in `langauge` attribute

    Default value for language is python

    Parameters
    ----------
    exression: Must use string.Template format for specifying the exression
        type: str
    language: default value is python. It could be any language. `evaluate` method expects the expression to be of type python

    Examples
    --------
    >>> exp = Expression("($x > 10 and $x <100) or ($x < -1 and $x > -500)")
    >>> exp.evaluate(x=500)
    False
    >>> exp.evaluate(x=20)
    True
    >>> exp.evaluate(x=9)
    False
    >>> exp.evaluate(x=-9)
    True
    """

    expression: str
    language: str = "python"

    def evaluate(self, **kwargs):
        if self.language.lower() != "python":
            raise Exception(
                f"Evaluation not supported for language ${self.language}. Supported language: python"
            )

        exp = Template(self.expression)
        final_expression = None
        try:
            final_expression = exp.substitute(kwargs)
        except:
            raise Exception(
                "Error substituting the value into the expression. Check if the variable in the expression matches the parameter names passed to evaluate method"
            )
        aeval = asteval.Interpreter()
        return aeval(final_expression)

    def __post_init__(self):
        if not self.expression or not self.language:
            raise ValueError(
                f"An Expression object requires values for both expression and language"
            )


@dataclass(repr=False)
class Domain(DataClassSerializable):
    """Domain describes the data. It holds following information -
    * stats - Statistics of the data.
    * constraints - List of Expression which defines the constraint for the data.
    * Domain values.

    Examples
    --------

    >>> Domain(values='Rational Numbers', stats={"mean":50, "median":51, "min": 5, "max":100}, constraints=[Expression('$x > 5')])
    constraints:
    - expression: $x > 5
        language: python
    stats:
        max: 100
        mean: 50
        median: 51
        min: 5
    values: Rational Numbers
    """

    values: str = ""
    stats: Dict = field(default_factory=dict)
    constraints: List[Expression] = field(default_factory=list)


@dataclass(repr=False, order=True)
class Attribute(DataClassSerializable):
    """
    Attribute describes the column/feature/element. It holds following information -
    * dtype - Type of data - float, int64, etc. Matches with Pandas dtypes
    * feature_type - Feature type of data - Integer, String, etc. Matches with ads feature types.
    * name - Name of the feature
    * domain - Represented by the Domain class
    * required - Boolean - True or False
    * description - Description about the column/feature
    * order - order of the column/feature in the data

    Examples
    --------

    >>> attr_fruits = Attribute(
    ...     dtype = "category",
    ...     feature_type = "category",
    ...     name = "fruits",
    ...     domain = Domain(values="Apple, Orange, Grapes", stats={"mode": "Orange"}, constraints=[Expression("in ['Apple', 'Orange', 'Grapes']")]),
    ...     required = True,
    ...     description = "Names of fruits",
    ...     order = 0
    ... )
    >>> attr_fruits
    description: Names of fruits
    domain:
        constraints:
        - expression: in ['Apple', 'Orange', 'Grapes']
            language: python
        stats:
            mode: Orange
        values: Apple, Orange, Grapes
    dtype: category
    feature_type: category
    name: fruits
    order: 0
    required: true
    >>> attr_fruits.key
    'fruits'
    """

    sort_index: int = field(init=False, repr=False)
    dtype: str
    feature_type: str
    name: str
    domain: Domain
    required: bool
    description: str
    order: Optional[int] = None

    @property
    def key(self):
        return self.name

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data.pop("sort_index", None)
        return data

    def __hash__(self):
        return hash(self.key)

    def __post_init__(self):
        object.__setattr__(self, "sort_index", self.order or 0)


class BaseSchemaLoader(ABC):
    """
    Base Schema Loader which load and validate schema.

    Methods
    -------
    load_schema(self)
        Load and validate schema from a file and return the normalized schema.
    """

    def __init__(self):
        self._schema = None

    def load_schema(self, schema_path):
        """Load and validate schema from a file and return the normalized schema."""
        self._load_schema(schema_path)
        self._normalize()
        return self._validate()

    def _normalize(self):
        self._schema = {key.lower(): value for key, value in self._schema.items()}

    @abstractmethod
    def _load_schema(self, schema_path):
        pass

    def _validate(self):
        """Validate the schema."""
        schema_validator = self._load_schema_validator()
        v = Validator(schema_validator)
        normalized_items = []
        for item in self._schema[SCHEMA_KEY]:
            valid = v.validate(item)
            if not valid:
                new_dict = {"column": item["name"], "error": v.errors}
                raise ValueError(json.dumps(new_dict, indent=2))
            normalized_items.append(v.normalized(item))
        schema_version = self._schema.get(SCHEMA_VERSION_KEY) or DEFAULT_SCHEMA_VERSION
        self._schema = {
            SCHEMA_KEY: normalized_items,
            SCHEMA_VERSION_KEY: schema_version,
        }
        return self._schema

    @staticmethod
    def _load_schema_validator():
        """load the schema validator to validate the schema."""
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), SCHEMA_VALIDATOR_NAME
            )
        ) as schema_file:
            schema_validator = json.load(schema_file)
        return schema_validator


class JsonSchemaLoader(BaseSchemaLoader):
    """
    Json Schema which load and validate schema from json file.

    Methods
    -------
    load_schema(self)
        Load and validate schema from json file and return the normalized schema.

    Examples
    --------
    >>> schema_loader = JsonSchemaLoader()
    >>> schema_dict = schema_loader.load_schema('schema.json')
    >>> schema_dict
    {'Schema': [{'dtype': 'object',
        'feature_type': 'String',
        'name': 'Attrition',
        'domain': {'values': 'String',
            'stats': {'count': 1470, 'unique': 2},
            'constraints': []},
        'required': True,
        'description': 'Attrition'},
        {'dtype': 'int64',
        'feature_type': 'Integer',
        'name': 'Age',
        'domain': {'values': 'Integer',
            'stats': {'count': 1470.0,
            'mean': 37.923809523809524,
            'std': 9.135373489136732,
            'min': 19.0,
            '25%': 31.0,
            '50%': 37.0,
            '75%': 44.0,
            'max': 61.0},
            'constraints': []},
        'required': True,
        'description': 'Age'}]}
    """

    def _load_schema(self, schema_path):
        """Loads and validates schema from a json file."""
        assert os.path.splitext(schema_path)[-1].lower() in [
            ".json"
        ], "Expecting a json format file."
        if not path.exists(schema_path):
            raise FileNotFoundError(f"{schema_path} does not exist")
        with fsspec.open(schema_path, mode="r", encoding="utf8") as f:
            self._schema = json.load(f)


class YamlSchemaLoader(BaseSchemaLoader):
    """
    Yaml Schema which loads and validates schema from a yaml file.

    Methods
    -------
    load_schema(self)
        Loads and validates schema from a yaml file and returns the normalized schema.

    Examples
    --------
    >>> schema_loader = YamlSchemaLoader()
    >>> schema_dict = schema_loader.load_schema('schema.yaml')
    >>> schema_dict
    {'Schema': [{'description': 'Attrition',
        'domain': {'constraints': [],
            'stats': {'count': 1470, 'unique': 2},
            'values': 'String'},
        'dtype': 'object',
        'feature_type': 'String',
        'name': 'Attrition',
        'required': True},
        {'description': 'Age',
        'domain': {'constraints': [],
            'stats': {'25%': 31.0,
            '50%': 37.0,
            '75%': 44.0,
            'count': 1470.0,
            'max': 61.0,
            'mean': 37.923809523809524,
            'min': 19.0,
            'std': 9.135373489136732},
            'values': 'Integer'},
        'dtype': 'int64',
        'feature_type': 'Integer',
        'name': 'Age',
        'required': True}]}
    """

    def _load_schema(self, schema_path):
        """Load and validate schema from yaml file."""
        assert os.path.splitext(schema_path)[-1].lower() in [
            ".yaml",
            ".yml",
        ], "Expecting yaml format file."
        if not path.exists(schema_path):
            raise FileNotFoundError(f"{schema_path} does not exist")
        with open(schema_path, "r") as stream:
            try:
                self._schema = yaml.load(yaml.safe_load(stream), Loader=loader)
            except yaml.YAMLError as exc:
                raise exc


class SchemaFactory:
    """
    Schema Factory.

    Methods
    -------
    register_format(self)
        Register a new type of schema class.
    get_schema(self)
        Get the YamlSchema or JsonSchema based on the format.
    default_schema(cls)
        Construct a SchemaFactory instance and register yaml and json loader.

    Examples
    --------
    >>> factory = SchemaFactory.default_schema()
    >>> schema_loader = factory.get_schema('.json')
    >>> schema_dict = schema_loader.load_schema('schema.json')
    >>> schema = Schema.from_dict(schema_dict)
    >>> schema
    Schema:
    - description: Attrition
    domain:
        constraints: []
        stats:
        count: 1470
        unique: 2
        values: String
    dtype: object
    feature_type: String
    name: Attrition
    required: true
    - description: Age
    domain:
        constraints: []
        stats:
        25%: 31.0
        50%: 37.0
        75%: 44.0
        count: 1470.0
        max: 61.0
        mean: 37.923809523809524
        min: 19.0
        std: 9.135373489136732
        values: Integer
    dtype: int64
    feature_type: Integer
    name: Age
    required: true
    """

    def __init__(self):
        self._creators = {}

    def register_format(self, file_format, creator):
        """Register a new type of schema class."""
        self._creators[file_format] = creator

    def get_schema(self, file_format):
        """Get the YamlSchema or JsonSchema based on the format."""
        creator = self._creators.get(file_format)
        if not creator:
            raise ValueError(
                f"This {file_format} format is not supported. Pass Json or Yaml Files."
            )
        return creator()

    @classmethod
    def default_schema(cls):
        factory = cls()
        factory.register_format(".json", JsonSchemaLoader)
        factory.register_format(".yaml", YamlSchemaLoader)
        factory.register_format(".yml", YamlSchemaLoader)
        return factory


@dataclass(repr=False)
class Schema:
    """
    Schema describes the structure of the data.

    Methods
    -------
    add(self, item: Attribute, replace: bool = False)
        Adds a new attribute item. Replaces existing one if replace flag is True.
    from_dict(self)
        Constructs an instance of Schema from a dictionary.
    from_file(cls, file_path):
        Loads the data schema from a file.
    to_dict(self)
        Serializes the data schema into a dictionary.
    to_yaml(self)
        Serializes the data schema into a YAML.
    to_json(self)
        Serializes the data schema into a json string.
    to_json_file(self)
        Saves the data schema into a json file.
    to_yaml_file(self)
        Save to a yaml file.
    add(self, item: Attribute, replace=False) -> None
        Adds a new attribute item. Replaces existing one if replace flag is True.

    Examples
    --------

    >>> attr_fruits = Attribute(
    ...     dtype = "category",
    ...     feature_type = "category",
    ...     name = "fruits",
    ...     domain = Domain(values="Apple, Orange, Grapes", stats={"mode": "Orange"}, constraints=[Expression("in ['Apple', 'Orange', 'Grapes']")]),
    ...     required = True,
    ...     description = "Names of fruits",
    ...     order = 0,
    ... )
    >>> attr_animals = Attribute(
    ...     dtype = "category",
    ...     feature_type = "category",
    ...     name = "animals",
    ...     domain = Domain(values="Dog, Cat, Python", stats={"mode": "Dog"}, constraints=[Expression("in ['Dog', 'Cat', 'Python']")]),
    ...     required = True,
    ...     description = "Names of animals",
    ...     order = 1,
    ... )
    >>> schema = Schema()
    >>> schema.add(attr_fruits)
    >>> schema.add(attr_animals)
    >>> schema
    schema:
    - description: Names of fruits
    domain:
        constraints:
        - expression: in ['Apple', 'Orange', 'Grapes']
        language: python
        stats:
        mode: Orange
        values: Apple, Orange, Grapes
    dtype: category
    feature_type: category
    name: fruits
    order: 0
    required: true
    - description: Names of animals
    domain:
        constraints:
        - expression: in ['Dog', 'Cat', 'Python']
        language: python
        stats:
        mode: Dog
        values: Dog, Cat, Python
    dtype: category
    feature_type: category
    name: animals
    order: 1
    required: true
    >>> schema.to_dict()
        {'schema': [{'dtype': 'category',
        'feature_type': 'category',
        'name': 'fruits',
        'domain': {'values': 'Apple, Orange, Grapes',
            'stats': {'mode': 'Orange'},
            'constraints': [{'expression': "in ['Apple', 'Orange', 'Grapes']",
            'language': 'python'}]},
        'required': True,
        'description': 'Names of fruits',
        'order': 0},
        {'dtype': 'category',
        'feature_type': 'category',
        'name': 'animals',
        'domain': {'values': 'Dog, Cat, Python',
            'stats': {'mode': 'Dog'},
            'constraints': [{'expression': "in ['Dog', 'Cat', 'Python']",
            'language': 'python'}]},
        'required': True,
        'description': 'Names of animals',
        'order': 1}]}

    """

    _schema: set = field(default_factory=set, init=False)
    _version: str = SCHEMA_VERSION

    def add(self, item: Attribute, replace: bool = False):
        """Adds a new attribute item. Replaces existing one if replace flag is True.

        Overrides the existing one if replace flag is True.

        Parameters
        ----------
        item : Attribute
            The attribute instance of a column/feature/element.
        replace : bool
            Overrides the existing attribute item if replace flag is True.

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
        if not isinstance(item, Attribute):
            raise TypeError("Argument must be an instance of the class Attribute.")
        if not replace and item.key in self.keys:
            raise ValueError(
                f"The key {item.key} already exists. Use `replace=True` to overwrite."
            )
        self._schema.discard(item)
        self._schema.add(item)

    @property
    def keys(self) -> list:
        """Returns all registered Attribute keys.

        Returns
        -------
        Tuple[str]
            The list of Attribute keys.
        """
        return tuple(item.key for item in self)

    @classmethod
    def from_dict(cls, schema: dict):
        """Constructs an instance of Schema from a dictionary.

        Parameters
        ----------
        schema : dict
            Data schema in dictionary format.

        Returns
        -------
        Schema
            An instance of Schema.
        """
        sc = cls()

        if schema == {}:
            return sc
        if not isinstance(schema, dict):
            raise TypeError("schema has to be of dictionary type.")

        schema = {key.lower(): value for key, value in deepcopy(schema).items()}
        for item in schema[SCHEMA_KEY]:
            domain = Domain(**item["domain"])
            domain.constraints = []
            for constraint in item["domain"]["constraints"]:
                domain.constraints.append(Expression(**constraint))
            item["domain"] = domain
            sc.add(Attribute(**item))
        return sc

    @classmethod
    def from_json(cls, schema: str):
        """Constructs an instance of Schema from a Json.

        Parameters
        ----------
        schema : str
            Data schema in Json format.

        Returns
        -------
        Schema
            An instance of Schema.
        """
        return Schema.from_dict(json.loads(schema))

    @classmethod
    def from_file(cls, file_path: str):
        """Loads the data schema from a file.

        Parameters
        ----------
        file_path : str
            File Path to load the data schema.

        Returns
        -------
        Schema
            An instance of Schema.
        """
        file_format = os.path.splitext(file_path)[-1]
        schema_loader = SchemaFactory.default_schema().get_schema(file_format)
        return cls.from_dict(schema_loader.load_schema(file_path))

    def to_dict(self):
        """Serializes data schema into a dictionary.

        Returns
        -------
        dict
            The dictionary representation of data schema.
        """
        return {
            SCHEMA_KEY: [item.to_dict() for item in self],
            SCHEMA_VERSION_KEY: self._version,
        }

    def to_yaml(self):
        """Serializes the data schema into a YAML.
        Returns
        -------
        str
            The yaml representation of data schema.
        """
        return yaml.dump(self.to_dict(), Dumper=dumper)

    def to_json(self):
        """Serializes the data schema into a json string.
        Returns
        -------
        str
            The json representation of data schema.
        """
        return json.dumps(self.to_dict()).replace("NaN", "null")

    def to_json_file(self, file_path, storage_options: dict = None):
        """Saves the data schema into a json file.

        Parameters
        ----------
        file_path : str
            File Path to store the schema in json format.
        storage_options: dict. Default None
            Parameters passed on to the backend filesystem class.
            Defaults to `storage_options` set using `DatasetFactory.set_default_storage()`.

        Returns
        -------
        None
            Nothing.
        """
        directory = os.path.expanduser(os.path.dirname(file_path))
        basename = os.path.expanduser(os.path.basename(file_path))
        assert os.path.splitext(file_path)[-1].lower() in [
            ".json"
        ], f"The file `{basename}` is not a valid JSON file. The `{file_path}` must have the extension .json."
        if directory and not os.path.exists(directory):
            if not ObjectStorageDetails.is_oci_path(directory):
                try:
                    os.mkdir(directory)
                except:
                    raise Exception(f"Error creating the directory.")
        if not storage_options:
            storage_options = DEFAULT_STORAGE_OPTIONS or {"config": {}}
        with fsspec.open(
            os.path.join(directory, basename),
            mode="w",
            **(storage_options),
        ) as f:
            f.write(json.dumps(self.to_dict()))

    def to_yaml_file(self, file_path):
        """Saves the data schema into a yaml file.
        Parameters
        ----------
        file_path : str
            File Path to store the schema in yaml format.

        Returns
        -------
        None
            Nothing.
        """
        assert os.path.splitext(file_path)[-1] in [
            ".yaml",
            ".yml",
        ], "The `file_path` must have the extension .yaml or .yml."
        directory = os.path.expanduser(os.path.dirname(file_path))
        basename = os.path.expanduser(os.path.basename(file_path))
        if directory and not os.path.exists(directory):
            try:
                os.mkdir(directory)
            except:
                raise Exception(f"Error creating the directory.")
        with open(os.path.join(directory, basename), "w") as yaml_file:
            yaml.dump(self.to_yaml(), yaml_file, default_flow_style=True)

    def validate_size(self) -> bool:
        """Validates schema size.

        Validates the size of schema. Throws an error if the size of the schema
        exceeds expected value.

        Returns
        -------
        bool
            True if schema does not exceeds the size limit.

        Raises
        ------
        SchemaSizeTooLarge
            If the size of the schema exceeds expected value.
        """
        if sys.getsizeof(self.to_yaml()) > INPUT_OUTPUT_SCHENA_SIZE_LIMIT:
            raise SchemaSizeTooLarge(sys.getsizeof(self.to_yaml()))
        return True

    def validate_schema(self):
        """Validate the schema."""
        schema_validator = BaseSchemaLoader._load_schema_validator()
        v = Validator(schema_validator)
        for item in self.to_dict()[SCHEMA_KEY]:
            valid = v.validate(item)
            if not valid:
                new_dict = {"column": item["name"], "error": v.errors}
                raise ValueError(json.dumps(new_dict, indent=2))
        return True

    def __getitem__(self, key: str):
        if key is None or key == "":
            raise ValueError(f"The key `{key}` must not be empty.")
        if not isinstance(key, str):
            TypeError(f"The key `{key}` must be a string.")
        for item in self._schema:
            if item.key == key:
                return item
        raise ValueError(f"The key {key} is not found.")

    def __repr__(self):
        return self.to_yaml()

    def __iter__(self):
        return sorted(self._schema).__iter__()

    def __len__(self):
        return len(self._schema)
