#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Union,
    TypeVar,
    Generic,
    Type,
    Optional,
    Tuple,
    DefaultDict,
)

import fsspec
from oci.resource_search.models import (
    ResourceSummaryCollection,
    StructuredSearchDetails,
)

from ads.common.oci_client import OCIClientFactory

from ads.dataset.progress import TqdmProgressBar

from ads.feature_store.entity import Entity
from ads.feature_store.feature_store import FeatureStore
from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.transformation import Transformation
import yaml

from ads.common import utils, auth

try:
    from yaml import CSafeLoader as Loader
except:
    from yaml import SafeLoader as Loader

logger = logging.getLogger(__name__)

_ModelBuilderT = TypeVar(
    "_ModelBuilderT",
    Type[FeatureStore],
    Type[Transformation],
    Type[FeatureGroup],
    Type[Entity],
    Type[Dataset],
)


class FeatureGroupMultipleTransformationsError(Exception):
    def __init__(self):
        super().__init__("A feature group cannot have more than one transformation")


class FeatureGroupMultipleEntitiesError(Exception):
    def __init__(self):
        super().__init__("A feature group cannot be linked to more than one entity")


class DataSetMultipleEntitiesError(Exception):
    def __init__(self):
        super().__init__("A dataset cannot be linked to more than one entity")


class _ModelBuilderHashDict(Generic[_ModelBuilderT]):
    def __init__(
        self,
        builders: Optional[List[_ModelBuilderT]],
        hash_fn: Callable = lambda model: model.name
        if model.attribute_map.get("name")
        else model.display_name,
    ):
        self.__hash_fn = hash_fn
        self._dict: Dict[str, _ModelBuilderT] = {}
        if builders is not None:
            for builder in builders:
                if builder is not None:
                    self.add(builder)

    def add(self, model: _ModelBuilderT) -> str:
        hash_ = self.__hash_fn(model)
        self._dict[hash_] = model
        return hash_

    def get_count(self):
        return len(self._dict.keys())

    def get_hash(self, model: _ModelBuilderT) -> str:
        return self.__hash_fn(model)

    def create_models(self, progress: TqdmProgressBar) -> List[_ModelBuilderT]:
        models = []
        item = 0
        for model in self._dict.values():
            if model is not None:
                item += 1
                progress.update(
                    "Creating {} {} of {}: {}".format(
                        model.kind, item, self.get_count(), self.get_hash(model)
                    ),
                    0,
                )
                if model.id is None:
                    models.append(model.create())
                else:
                    models.append(model)
                progress.update(n=1)
        return models

    def get_dict(self):
        return self._dict

    def get(self, key: str) -> Union[None, _ModelBuilderT]:
        return self._dict.get(key)


class _ElementMap(Generic[_ModelBuilderT]):
    def __init__(
        self,
        element_type: _ModelBuilderT,
        element_dict: _ModelBuilderHashDict[_ModelBuilderT],
        parent_child_map: Dict[str, "_ParentChildMap"] = None,
    ):
        self.element_type = element_type
        self.element_dict = element_dict
        self.parent_child_map = parent_child_map

    def add_element_from_dict(self, config: dict) -> str:
        return self.element_dict.add(self.element_type.from_dict(config))


class _ParentChildMap:
    def __init__(
        self,
        child_mapping: "_ElementMap",
        parent_child_hash_map: DefaultDict[str, List[str]],
    ):
        self.child_mapping = child_mapping
        self.parent_child_hash_map = parent_child_hash_map


class FeatureStoreRegistrar:
    YAML_FILE_PATH = os.path.join(
        os.path.dirname(__file__), "templates/feature_store_template.yaml"
    )
    TRANSFORMATION_SPEC = "transformation"
    ENTITY_SPEC = "entity"
    FEATURE_GROUP_SPEC = "featureGroup"
    DATASET_SPEC = "dataset"

    def __init__(
        self,
        feature_store: FeatureStore = None,
        entities: List[Entity] = None,
        datasets: List[Dataset] = None,
        feature_groups: List[FeatureGroup] = None,
        transformations: List[Transformation] = None,
    ):
        """
        Initialised feature registrar resource
        Parameters
        ----------
        :type transformations: Transformation models to be created
        :type feature_groups: Feature group models to be created
        :type datasets: Dataset models to be created
        :type entities: Entities to be created
        :type feature_store: FeatureStore to be created
        """
        self._feature_store_id = None
        self._root_compartment_id = None
        self._feature_store = _ModelBuilderHashDict([feature_store] or [])
        self._entities = _ModelBuilderHashDict(entities)
        self._datasets = _ModelBuilderHashDict(datasets)
        self._feature_groups = _ModelBuilderHashDict(feature_groups)
        self._transformations = _ModelBuilderHashDict(transformations)
        self.feature_group_transformation_map = defaultdict(lambda: [])
        self.feature_group_entity_map = defaultdict(lambda: [])
        self.dataset_entity_map = defaultdict(lambda: [])
        self._progress = None
        self._entity_map = {}

    def create(
        self,
    ) -> Tuple[
        FeatureStore,
        List[Entity],
        List[Transformation],
        List[FeatureGroup],
        List[Dataset],
    ]:
        """
        Creates feature store resources

        Returns
        -------
        Tuple
            Tuple containing feature store resources
        """
        self._progress = utils.get_progress_bar(
            max_progress=self._get_progress_steps_count()
        )
        feature_store = self._create_feature_store()
        self._feature_store_id = feature_store.id
        entities = self._create_entities()
        transformations = self._create_transformations()
        feature_groups = self._create_feature_groups()
        datasets = self._create_datasets()
        print(
            f"Successfully created {len(entities)} entities, {len(transformations)} transformations, {len(feature_groups)} feature groups and {len(datasets)} datasets"
        )
        return (
            feature_store,
            entities,
            transformations,
            feature_groups,
            datasets,
        )

    def _create_feature_store(self) -> FeatureStore:
        feature_store = self._feature_store.create_models(self._progress)
        if len(feature_store) == 1:
            self._root_compartment_id = feature_store[0].compartment_id
            return feature_store[0]
        elif len(feature_store) > 1:
            raise ValueError("Only one feature store can be associated with registrar")
        else:
            raise ValueError("Unable to create feature store resource")

    def _create_entities(self) -> List[Entity]:
        for entity in self._entities.get_dict().values():
            entity.feature_store_id = self._feature_store_id
            entity.compartment_id = entity.compartment_id or self._root_compartment_id
        entities = self._entities.create_models(self._progress)
        for entity in entities:
            self._entity_map[entity.name] = entity.id
        return entities

    def _create_transformations(self) -> List[Transformation]:
        for transformation in self._transformations.get_dict().values():
            transformation.feature_store_id = self._feature_store_id
            transformation.compartment_id = (
                transformation.compartment_id or self._root_compartment_id
            )
            # to encode to base64
            transformation.source_code_function = transformation.source_code_function
        return self._transformations.create_models(self._progress)

    def _create_feature_groups(self) -> List[FeatureGroup]:
        for feature_group in self._feature_groups.get_dict().values():
            linked_transformations: List[str] = self.feature_group_transformation_map[
                self._feature_groups.get_hash(feature_group)
            ]
            if feature_group.transformation_id is None:
                if len(linked_transformations) == 1:
                    feature_group.transformation_id = self._transformations.get(
                        linked_transformations[0]
                    ).oci_fs_transformation.id
                elif len(linked_transformations) > 1:
                    raise FeatureGroupMultipleTransformationsError()
            elif len(linked_transformations) > 0:
                raise FeatureGroupMultipleTransformationsError()

            linked_entity: List[str] = self.feature_group_entity_map[
                self._feature_groups.get_hash(feature_group)
            ]
            if feature_group.entity_id is None:
                if len(linked_entity) == 1:
                    feature_group.entity_id = self._entities.get(
                        linked_entity[0]
                    ).oci_fs_entity.id
                elif len(linked_entity) > 1:
                    raise FeatureGroupMultipleEntitiesError()
            elif len(linked_entity) > 0:
                raise FeatureGroupMultipleEntitiesError()

            # to put primary keys in array
            feature_group.primary_keys = feature_group.primary_keys
            feature_group.feature_store_id = self._feature_store_id
            feature_group.compartment_id = (
                feature_group.compartment_id or self._root_compartment_id
            )

        return self._feature_groups.create_models(self._progress)

    def _create_datasets(self) -> List[Dataset]:
        for dataset in self._datasets.get_dict().values():
            linked_entity: List[str] = self.dataset_entity_map[
                self._datasets.get_hash(dataset)
            ]
            if dataset.entity_id is None:
                if len(linked_entity) == 1:
                    dataset.entity_id = self._entities.get(
                        linked_entity[0]
                    ).oci_fs_entity.id
                elif len(linked_entity) > 1:
                    raise DataSetMultipleEntitiesError()
            elif len(linked_entity) > 1:
                raise DataSetMultipleEntitiesError()
            dataset.feature_store_id = self._feature_store_id
            dataset.compartment_id = dataset.compartment_id or self._root_compartment_id
            original_query = dataset.query
            for fg_name, entity_name in self.feature_group_entity_map.items():
                dataset.query = dataset.query.replace(
                    f"{entity_name[0]}.{fg_name}",
                    f"`{self._entity_map[entity_name[0]]}`.{fg_name}",
                )
            if original_query == dataset.query:
                # TODO: remove dataset from list
                pass
                # invalid_dataset = self._datasets.get_dict()
                # del invalid_dataset[dataset.name]
                # print("Invalid query, dataset won't be created")
        return self._datasets.create_models(self._progress)

    @classmethod
    def from_dict(cls, config: dict) -> "FeatureStoreRegistrar":
        """Initialize the object from a Python dictionary
        Args:
           config (dict): dictionary containing definition for feature store design time entities
        """
        if not isinstance(config, dict) or not config:
            raise ValueError(
                "The config data for initializing feature store registrar is invalid."
            )
        registrar = cls()

        entity_map = _ElementMap(Entity, registrar._entities)
        transformation_map = _ElementMap(Transformation, registrar._transformations)
        feature_group_map = _ElementMap(
            FeatureGroup,
            registrar._feature_groups,
            {
                registrar.TRANSFORMATION_SPEC: _ParentChildMap(
                    transformation_map, registrar.feature_group_transformation_map
                ),
                registrar.ENTITY_SPEC: _ParentChildMap(
                    entity_map, registrar.feature_group_entity_map
                ),
            },
        )
        dataset_map = _ElementMap(
            Dataset,
            registrar._datasets,
            {
                registrar.ENTITY_SPEC: _ParentChildMap(
                    entity_map, registrar.dataset_entity_map
                )
            },
        )

        mappings: Dict[str, _ElementMap] = {
            cls.ENTITY_SPEC: entity_map,
            cls.TRANSFORMATION_SPEC: transformation_map,
            cls.FEATURE_GROUP_SPEC: feature_group_map,
            cls.DATASET_SPEC: dataset_map,
        }

        registrar._feature_store = _ModelBuilderHashDict(
            [FeatureStore.from_dict(config)]
        )
        spec = config.get("spec")

        for key, elements in spec.items():
            mapping = mappings.get(key)
            if mapping is not None:
                for element in elements:
                    hash_ = mapping.add_element_from_dict(element)
                    if "spec" in element and mapping.parent_child_map is not None:
                        cls._populate_child_mappings(element, mapping, hash_)

        return registrar

    @classmethod
    def generate_yaml(cls, uri: str = "feature_store.yaml"):
        """Generates YAML scaffolding for setting up a feature store
        Args:
            uri (string): URI location for feature store YAML
        """
        template = cls._read_from_file(uri=cls.YAML_FILE_PATH)
        template = cls._set_metastore_id(template)
        with fsspec.open(uri, "w") as op_file:
            op_file.write(template)

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str = None,
        uri: str = None,
        loader: callable = Loader,
        **kwargs,
    ) -> "FeatureStoreRegistrar":
        """Creates an object from YAML string provided or from URI location containing YAML string

        Args:
            yaml_string (string, optional): YAML string. Defaults to None.
            uri (string, optional): URI location of file containing YAML string. Defaults to None.
            loader (callable, optional): Custom YAML loader. Defaults to CLoader/SafeLoader.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be
            config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Raises:
            ValueError: Raised if neither string nor uri is provided

        Returns:
            cls: Returns instance of the class
        """
        if yaml_string:
            return cls.from_dict(yaml.load(yaml_string, Loader=loader))
        if uri is None:
            uri = cls._find_yaml_definition_file()
        if uri:
            return cls.from_dict(
                yaml.load(cls._read_from_file(uri=uri, **kwargs), Loader=loader)
            )
        else:
            raise ValueError(
                "Unable to find yaml definition file, must provide either YAML string or URI location"
            )

    @staticmethod
    def _find_yaml_definition_file() -> str:
        uri = None
        # get the current working directory
        cwd = os.getcwd()
        logger.info("scanning current directory for yaml definition file")
        for file_name in os.listdir(cwd):
            if file_name.endswith(".yaml") or file_name.endswith(".yml"):
                if uri is None:
                    uri = f"{cwd}/{file_name}"
                else:
                    return None
        return uri

    def _get_progress_steps_count(self) -> int:
        return (
            self._feature_store.get_count()
            + self._entities.get_count()
            + self._transformations.get_count()
            + self._feature_groups.get_count()
            + self._datasets.get_count()
        )

    @staticmethod
    def _read_from_file(uri: str, **kwargs) -> str:
        """Returns contents from location specified by URI

        Args:
            uri (string): URI location
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be
            config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Returns:
            string: Contents in file specified by URI
        """
        with fsspec.open(uri, "r", **kwargs) as file:
            return file.read()

    @staticmethod
    def _populate_child_mappings(
        parent_dict: dict, parent_map: _ElementMap, parent_hash: str
    ):
        for key, elements in parent_dict.get("spec").items():
            if key in parent_map.parent_child_map:
                parent_child_mapping = parent_map.parent_child_map[key]
                if isinstance(elements, list):
                    for element in elements:
                        child_hash = (
                            parent_child_mapping.child_mapping.add_element_from_dict(
                                element
                            )
                        )
                        parent_child_mapping.parent_child_hash_map[parent_hash].append(
                            child_hash
                        )
                else:
                    child_hash = (
                        parent_child_mapping.child_mapping.add_element_from_dict(
                            elements
                        )
                    )
                    parent_child_mapping.parent_child_hash_map[parent_hash].append(
                        child_hash
                    )

    @staticmethod
    def _set_metastore_id(template: str) -> str:
        resource_client = OCIClientFactory(**auth.default_signer()).create_client(
            "resource_search"
        )
        try:
            resource_collection: ResourceSummaryCollection = (
                resource_client.search_resources(
                    StructuredSearchDetails(
                        query="query datacatalogmetastore resources"
                    )
                ).data
            )
            if len(resource_collection.items) == 1:
                # populate if there is only 1 metastore in tenancy
                template = template.replace(
                    "{metastore_id}", resource_collection.items[0].identifier
                )

        except Exception as e:
            logger.warning("Unable to get metastoreid due to: ", e, exc_info=True)
        finally:
            return template
