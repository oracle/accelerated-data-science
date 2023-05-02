#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import contextlib
import copy
import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Union

import oci.data_science
from ads.common.utils import batch_convert_case, get_value, snake_to_camel
from ads.config import COMPARTMENT_OCID, OCI_REGION_METADATA, PROJECT_OCID
from ads.jobs.builders.base import Builder
from ads.model.datascience_model import DataScienceModel
from ads.model.service.oci_datascience_model_version_set import (
    DataScienceModelVersionSet,
    ModelVersionSetNotExists,
    ModelVersionSetNotSaved,
)
from oci.data_science.models import UpdateModelDetails

logger = logging.getLogger(__name__)

_MVS_ID_ENV_VAR = "MODEL_VERSION_SET_ID"
_MVS_NAME_ENV_VAR = "MODEL_VERSION_SET_NAME"
_MVS_COMPARTMENT_ENV_VAR = "MODEL_VERSION_SET_COMPARTMENT_ID"
_MVS_URL = (
    "https://console.{region}.oraclecloud.com/data-science/model-version-sets/{id}"
)


class ModelVersionSet(Builder):
    """Represents Model Version Set.

    Attributes
    ----------
    id: str
        Model version set OCID.
    project_id: str
        Project OCID.
    compartment_id: str
        Compartment OCID.
    name: str
        Model version set name.
    description: str
        Model version set description.
    freeform_tags: Dict[str, str]
        Model version set freeform tags.
    defined_tags: Dict[str, Dict[str, object]]
        Model version set defined tags.
    details_link: str
        Link to details page in OCI console.

    Methods
    -------
    create(self, **kwargs) -> "ModelVersionSet"
        Creates a model version set.
    update(self, **kwargs) -> "ModelVersionSet"
        Updates a model version set.
    delete(self, delete_model: Optional[bool] = False) -> "ModelVersionSet":
        Removes a model version set.
    to_dict(self) -> dict
        Serializes model version set to a dictionary.
    from_id(cls, id: str) -> "ModelVersionSet"
        Gets an existing model version set by OCID.
    from_ocid(cls, ocid: str) -> "ModelVersionSet"
        Gets an existing model version set by OCID.
    from_name(cls, name: str) -> "ModelVersionSet"
        Gets an existing model version set by name.
    from_dict(cls, config: dict) -> "ModelVersionSet"
        Load a model version set instance from a dictionary of configurations.

    Examples
    --------
    >>> mvs = (ModelVersionSet()
    ...    .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    ...    .with_project_id(os.environ["PROJECT_OCID"])
    ...    .with_name("test_experiment")
    ...    .with_description("Experiment number one"))
    >>> mvs.create()
    >>> mvs.model_add(model_ocid, version_label="Version label 1")
    >>> mvs.model_list()
    >>> mvs.details_link
    ... https://console.<region>.oraclecloud.com/data-science/model-version-sets/<ocid>
    >>> mvs.delete()
    """

    LIFECYCLE_STATE_ACTIVE = "ACTIVE"
    LIFECYCLE_STATE_DELETING = "DELETING"
    LIFECYCLE_STATE_DELETED = "DELETED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    CONST_ID = "id"
    CONST_PROJECT_ID = "projectId"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"

    attribute_map = {
        CONST_ID: "id",
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes a model version set.

        Parameters
        ----------
        spec: (Dict, optional). Defaults to None.
            Object specification.

        kwargs: Dict
            Specification as keyword arguments.
            If 'spec' contains the same key as the one in kwargs,
            the value from kwargs will be used.

            - project_id: str
            - compartment_id: str
            - name: str
            - description: str
            - defined_tags: Dict[str, Dict[str, object]]
            - freeform_tags: Dict[str, str]
        """
        defaults = ModelVersionSet._load_default_properties()
        if spec:
            spec = {
                snake_to_camel(k): v
                for k, v in spec.items()
                if snake_to_camel(k) in self.attribute_map and v is not None
            }
            defaults.update(spec)

        spec = {
            snake_to_camel(k): v
            for k, v in kwargs.items()
            if snake_to_camel(k) in self.attribute_map and v is not None
        }
        defaults.update(spec)

        super().__init__(spec=defaults)
        self.dsc_model_version_set = DataScienceModelVersionSet(**self._spec)

    @property
    def project_id(self) -> str:
        return self.get_spec(self.CONST_PROJECT_ID)

    @project_id.setter
    def project_id(self, value: str):
        self.with_project_id(value)

    def with_project_id(self, project_id: str) -> "ModelVersionSet":
        """Sets the project OCID.

        Parameters
        ----------
        project_id: str
            The project OCID.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "ModelVersionSet":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the model version set.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)

        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "ModelVersionSet":
        """Sets the compartment OCID.

        Parameters
        ----------
        compartment_id: str
            The compartment OCID.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, value: str):
        self.with_name(value)

    def with_name(self, name: str) -> "ModelVersionSet":
        """Sets the name of the model version set.

        Parameters
        ----------
        name: str
            The name of the model version set.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def freeform_tags(self) -> Dict[str, str]:
        return self.get_spec(self.CONST_FREEFORM_TAG)

    @freeform_tags.setter
    def freeform_tags(self, value: Dict[str, str]):
        self.with_freeform_tags(**value)

    def with_freeform_tags(self, **kwargs: Dict[str, str]) -> "ModelVersionSet":
        """Sets freeform tags.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        return self.set_spec(self.CONST_FREEFORM_TAG, kwargs)

    @property
    def defined_tags(self) -> Dict[str, Dict[str, object]]:
        return self.get_spec(self.CONST_FREEFORM_TAG)

    @defined_tags.setter
    def defined_tags(self, value: Dict[str, Dict[str, object]]):
        self.with_defined_tags(**value)

    def with_defined_tags(
        self, **kwargs: Dict[str, Dict[str, object]]
    ) -> "ModelVersionSet":
        """Sets defined tags.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        return self.set_spec(self.CONST_DEFINED_TAG, kwargs)

    def create(self, **kwargs) -> "ModelVersionSet":
        """Creates a model version set.

        Parameters
        ----------
        kwargs
            Additional keyword arguments.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self)
        """
        if not self.name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.name = f"model-version-set-{timestamp}"

        payload = copy.deepcopy(self._spec)
        payload.update({"name": self.name})

        if not payload.get(self.CONST_COMPARTMENT_ID, None):
            raise ValueError(
                "Compartment id is required. Specify compartment id via `with_compartment_id()`."
            )

        payload.pop("id", None)
        logger.debug(f"Creating a model version set with payload {payload}")
        self.dsc_model_version_set = DataScienceModelVersionSet(**payload).create()
        # self.with_id(self.dsc_model_version_set.id)
        self._update_from_dsc_model(self.dsc_model_version_set)
        return self

    def update(self) -> "ModelVersionSet":
        """Updates a model version set.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self).
        """
        payload = copy.deepcopy(self._spec)
        logger.debug(f"Updating a model version set with payload {payload}")
        self.dsc_model_version_set = DataScienceModelVersionSet(**payload).update()
        self._update_from_dsc_model(self.dsc_model_version_set)
        return self

    def delete(self, delete_model: Optional[bool] = False) -> "ModelVersionSet":
        """Removes a model version set.

        Parameters
        ----------
        delete_model: (bool, optional). Defaults to False.
            By default, this parameter is false. A model version set can only be
            deleted if all the models associate with it are already in the DELETED state.
            You can optionally specify the deleteRelatedModels boolean query parameters to
            true, which deletes all associated models for you.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self).
        """
        self.dsc_model_version_set.delete(delete_model)

    def model_add(
        self, model_id: str, version_label: Optional[str] = None, **kwargs
    ) -> None:
        """Adds new model to model version set.

        Parameters
        ----------
        model_id: str
            The OCID of the model which needs to be associated with the model version set.
        version_label: str
            The model version label.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ModelVersionSetNotSaved: If model version set has not been saved yet.
        """
        if not self.id:
            raise ModelVersionSetNotSaved("Model version set needs to be saved.")

        self.dsc_model_version_set.client.update_model(
            model_id,
            UpdateModelDetails(
                model_version_set_id=self.id, version_label=version_label
            ),
            **kwargs,
        )

    @classmethod
    def _load_default_properties(self) -> Dict:
        """
        Load default properties from environment variables, notebook session, etc.

        Returns
        -------
        Dict
            A dictionary of default properties.
        """
        defaults = {}

        if COMPARTMENT_OCID:
            defaults[self.CONST_COMPARTMENT_ID] = COMPARTMENT_OCID
        if PROJECT_OCID:
            defaults[self.CONST_PROJECT_ID] = PROJECT_OCID

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        defaults[self.CONST_NAME] = f"model-version-set-{timestamp}"

        return defaults

    @property
    def id(self) -> Optional[str]:
        """The OCID of the model version set."""
        if self.dsc_model_version_set:
            return self.dsc_model_version_set.id
        return None

    @property
    def status(self) -> Optional[str]:
        """Status of the model version set.

        Returns
        -------
        str
            Status of the model version set.
        """
        if self.dsc_model_version_set:
            return self.dsc_model_version_set.status
        return None

    @property
    def details_link(self) -> str:
        """Link to details page in OCI console.

        Returns
        -------
        str
            Link to details page in OCI console.
        """
        if self.dsc_model_version_set:
            signer = self.dsc_model_version_set.auth or {}
            if "region" in signer.get("config", {}):
                region = signer["config"]["region"]
            else:
                region = json.loads(OCI_REGION_METADATA)["regionIdentifier"]
            return _MVS_URL.format(region=region, id=self.id)

        return None

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            "modelVersionSet"
        """
        return "modelVersionSet"

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["ModelVersionSet"]:
        """
        List model version sets in a given compartment.

        Parameters
        ----------
        compartment_id: str
            The OCID of compartment.
        kwargs
            Additional keyword arguments for filtering model version sets.

        Returns
        -------
        List[ModelVersionSet]
            The list of model version sets.
        """
        return [
            cls.from_dsc_model_version_set(model_version_set)
            for model_version_set in DataScienceModelVersionSet.list_resource(
                compartment_id, **kwargs
            )
        ]

    def models(self, **kwargs) -> List[DataScienceModel]:
        """Gets list of models associated with a model version set.

        Parameters
        ----------
        kwargs:
            project_id: str
                Project OCID.
           lifecycle_state: str
                Filter results by the specified lifecycle state. Must be a valid state for the resource type.
                Allowed values are: "ACTIVE", "DELETED", "FAILED", "INACTIVE"

            Can be any attribute that `oci.data_science.data_science_client.DataScienceClient.list_models.` accepts.

        Returns
        -------
        List[DataScienceModel]
            List of models associated with the model version set.

        Raises
        ------
        ModelVersionSetNotSaved: If model version set has not been saved yet.
        """
        if not self.id:
            raise ModelVersionSetNotSaved("Model version set needs to be saved.")

        return DataScienceModel.list(
            compartment_id=self.compartment_id,
            model_version_set_name=self.name,
            **kwargs,
        )

    @classmethod
    def from_dsc_model_version_set(
        cls, dsc_model_version_set: DataScienceModelVersionSet
    ) -> "ModelVersionSet":
        """Initialize a ModelVersionSet instance from a DataScienceModelVersionSet.

        Parameters
        ----------
        dsc_model_version_set: DataScienceModelVersionSet
            An instance of DataScienceModelVersionSet.

        Returns
        -------
        ModelVersionSet
            An instance of ModelVersionSet.
        """
        return cls()._update_from_dsc_model(dsc_model_version_set)

    def _update_from_dsc_model(
        self, dsc_model_version_set: oci.data_science.models.ModelVersionSet
    ) -> "ModelVersionSet":
        """Update the properties from an OCI data science model version set model.

        Parameters
        ----------
        dsc_model_version_set: oci.data_science.models.ModelVersionSet
            An OCI data science model version set model.

        Returns
        -------
        ModelVersionSet
            The ModelVersionSet instance (self).
        """
        self.dsc_model_version_set = dsc_model_version_set
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = get_value(dsc_model_version_set, dsc_attr)
            if value:
                self._spec[infra_attr] = get_value(dsc_model_version_set, dsc_attr)
        return self

    @classmethod
    def from_id(cls, id: str) -> "ModelVersionSet":
        """Gets an existing model version set by OCID.

        Parameters
        ----------
        id: str
            The model version set OCID.

        Returns
        -------
        ModelVersionSet
            An instance of ModelVersionSet.
        """
        return cls.from_dsc_model_version_set(DataScienceModelVersionSet.from_ocid(id))

    @classmethod
    def from_ocid(cls, ocid: str) -> "ModelVersionSet":
        """Gets an existing model version set by OCID.

        Parameters
        ----------
        id: str
            The model version set OCID.

        Returns
        -------
        ModelVersionSet
            An instance of ModelVersionSet.
        """
        return cls.from_id(ocid)

    @classmethod
    def from_name(
        cls, name: str, compartment_id: Optional[str] = None
    ) -> "ModelVersionSet":
        """Gets an existing model version set by name.

        Parameters
        ----------
        name: str
            The model version set name.
        compartment_id: (str, optional). Defaults to None.
            Compartment OCID of the OCI resources. If `compartment_id` is not specified,
            the value will be taken from environment variables.

        Returns
        -------
        ModelVersionSet
            An instance of ModelVersionSet.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        return cls.from_dsc_model_version_set(
            DataScienceModelVersionSet.from_name(
                name=name, compartment_id=compartment_id
            )
        )

    def to_dict(self) -> dict:
        """
        Serializes model version set to a dictionary.

        Returns
        -------
        dict
            The model version set serialized as a dictionary.
        """
        return {
            "kind": self.kind,
            "type": self.type,
            "spec": batch_convert_case(self._spec, "camel"),
        }

    @classmethod
    def from_dict(cls, config: dict) -> "ModelVersionSet":
        """
        Load a model version set instance from a dictionary of configurations.

        Parameters
        ----------
        config: dict
            A dictionary of configurations.

        Returns
        -------
        ModelVersionSet
            The model version set instance.
        """
        return cls(spec=batch_convert_case(config.get("spec"), "snake"))

    def __getattr__(self, item):
        if f"with_{item}" in self.__dir__():
            return self.get_spec(item)
        raise AttributeError(f"Attribute {item} not found.")


def _extract_model_version_set_id(
    model_version_set: Optional[Union[str, ModelVersionSet]] = None,
    compartment_id: Optional[str] = None,
) -> str:
    """Extracts model version set id.
    If `model_version_set` attribute not specified, then the `model_version_set_id`
    will be extracted from the environment variables.

    Parameters
    ----------
    model_version_set: (Union[str, ModelVersionSet], optional). Defaults to None.
        The model version set information. Can be name, ocid or `ModelVersionSet` instance.
        If `model_version_set` not provided then id will be extracted from environment variables.
    compartment_id: (str, optional). Defaults to value from the environment variables.
            The compartment OCID.

    Returns
    -------
    str
        The model version set OCID.
    """
    compartment_id = (
        compartment_id or os.environ.get(_MVS_COMPARTMENT_ENV_VAR) or COMPARTMENT_OCID
    )
    if not model_version_set:
        return os.environ.get(_MVS_ID_ENV_VAR)
    if isinstance(model_version_set, ModelVersionSet):
        return model_version_set.id
    if isinstance(model_version_set, str):
        if model_version_set.lower().startswith("ocid"):
            return model_version_set
        # search model version set by name
        return ModelVersionSet.from_name(
            name=model_version_set, compartment_id=compartment_id
        ).id
    return None


@contextlib.contextmanager
def experiment(name: str, create_if_not_exists: Optional[bool] = True, **kwargs: Dict):
    """Context manager helping to operate with model version set.

    Parameters
    ----------
    name: str
        The name of the model version set.
    create_if_not_exists: (bool, optional). Defaults to True.
        Creates model version set if not exists.

    kwargs: (Dict, optional).
        compartment_id: (str, optional). Defaults to value from the environment variables.
            The compartment OCID.
        project_id: (str, optional). Defaults to value from the environment variables.
            The project OCID.
        description: (str, optional). Defaults to None.
            The description of the model version set.

    Yields
    ------
    ModelVersionSet
        The model version set object.
    """
    compartment_id = kwargs.pop("compartment_id", None) or COMPARTMENT_OCID
    project_id = kwargs.pop("project_id", None) or PROJECT_OCID

    try:
        mvs = ModelVersionSet.from_name(name=name, compartment_id=compartment_id)
    except ModelVersionSetNotExists:
        if not create_if_not_exists:
            raise
        else:
            mvs = (
                ModelVersionSet()
                .with_compartment_id(compartment_id)
                .with_project_id(project_id)
                .with_name(name)
                .with_description(kwargs.pop("description", ""))
                .create()
            )

    try:
        os.environ[_MVS_ID_ENV_VAR] = mvs.id
        os.environ[_MVS_NAME_ENV_VAR] = mvs.name
        os.environ[_MVS_COMPARTMENT_ENV_VAR] = compartment_id

        yield mvs
    finally:
        os.environ.pop(_MVS_ID_ENV_VAR, None)
        os.environ.pop(_MVS_NAME_ENV_VAR, None)
        os.environ.pop(_MVS_COMPARTMENT_ENV_VAR, None)
