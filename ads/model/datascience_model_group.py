#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
from typing import Dict, List, Union

from ads.common.utils import batch_convert_case
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.jobs.builders.base import Builder
from ads.model.model_metadata import ModelCustomMetadata
from ads.model.service.oci_datascience_model_group import OCIDataScienceModelGroup

try:
    from oci.data_science.models import (
        CreateModelGroupDetails,
        CustomMetadata,
        HomogeneousModelGroupDetails,
        MemberModelDetails,
        MemberModelEntries,
        ModelGroup,
        ModelGroupDetails,
        ModelGroupSummary,
        StackedModelGroupDetails,
        UpdateModelGroupDetails,
    )
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "The oci model group module was not found. Please run `pip install oci` "
        "to install the latest oci sdk."
    ) from err

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
ALLOWED_CREATE_TYPES = ["CREATE", "CLONE"]
MODEL_GROUP_KIND = "datascienceModelGroup"


class DataScienceModelGroup(Builder):
    """Represents a Data Science Model Group.

    Attributes
    ----------
    id: str
        Model group ID.
    project_id: str
        Project OCID.
    compartment_id: str
        Compartment OCID.
    display_name: str
        Model group name.
    description: str
        Model group description.
    freeform_tags: Dict[str, str]
        Model group freeform tags.
    defined_tags: Dict[str, Dict[str, object]]
        Model group defined tags.
    custom_metadata_list: ModelCustomMetadata
        Model group custom metadata.
    model_group_version_history_name: str
        Model group version history name
    model_group_version_history_id: str
        Model group version history ID
    version_label: str
        Model group version label
    version_id: str
        Model group version id
    lifecycle_state: str
        Model group lifecycle state
    lifecycle_details: str
        Model group lifecycle details

    Methods
    -------
    activate(self, ...) -> "DataScienceModelGroup"
        Activates model group.
    create(self, ...) -> "DataScienceModelGroup"
        Creates model group.
    deactivate(self, ...) -> "DataScienceModelGroup"
        Deactivates model group.
    delete(self, ...) -> "DataScienceModelGroup":
        Deletes model group.
    to_dict(self) -> dict
        Serializes model group to a dictionary.
    from_id(cls, id: str) -> "DataScienceModelGroup"
        Gets an existing model group by OCID.
    from_dict(cls, config: dict) -> "DataScienceModelGroup"
        Loads model group instance from a dictionary of configurations.
    update(self, ...) -> "DataScienceModelGroup"
        Updates datascience model group in model catalog.
    list(cls, compartment_id: str = None, **kwargs) -> List["DataScienceModelGroup"]
        Lists datascience model groups in a given compartment.
    sync(self):
        Sync up a datascience model group with OCI datascience model group.
    with_project_id(self, project_id: str) -> "DataScienceModelGroup"
        Sets the project ID.
    with_description(self, description: str) -> "DataScienceModelGroup"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "DataScienceModelGroup"
        Sets the compartment ID.
    with_display_name(self, name: str) -> "DataScienceModelGroup"
        Sets the name.
    with_freeform_tags(self, **kwargs: Dict[str, str]) -> "DataScienceModelGroup"
        Sets freeform tags.
    with_defined_tags(self, **kwargs: Dict[str, Dict[str, object]]) -> "DataScienceModelGroup"
        Sets defined tags.
    with_custom_metadata_list(self, metadata: Union[ModelCustomMetadata, Dict]) -> "DataScienceModelGroup"
        Sets model group custom metadata.
    with_model_group_version_history_id(self, model_group_version_history_id: str) -> "DataScienceModelGroup":
        Sets the model group version history ID.
    with_version_label(self, version_label: str) -> "DataScienceModelGroup":
        Sets the model group version label.
    with_base_model_id(self, base_model_id) -> "DataScienceModelGroup":
        Sets the base model ID.
    with_member_models(self, member_models: List[Dict[str, str]]) -> "DataScienceModelGroup":
        Sets the list of member models to be grouped.


    Examples
    --------
    >>> ds_model_group = (DataScienceModelGroup()
    ...    .with_compartment_id(os.environ["NB_SESSION_COMPARTMENT_OCID"])
    ...    .with_project_id(os.environ["PROJECT_OCID"])
    ...    .with_display_name("TestModelGroup")
    ...    .with_description("Testing the model group")
    ...    .with_freeform_tags(tag1="val1", tag2="val2")
    >>> ds_model_group.create()
    >>> ds_model_group.with_description("new description").update()
    >>> ds_model_group.delete()
    >>> DataScienceModelGroup.list()
    """

    CONST_ID = "id"
    CONST_CREATE_TYPE = "createType"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_PROJECT_ID = "projectId"
    CONST_DISPLAY_NAME = "displayName"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_MODEL_GROUP_DETAILS = "modelGroupDetails"
    CONST_MEMBER_MODEL_ENTRIES = "memberModelEntries"
    CONST_CUSTOM_METADATA_LIST = "customMetadataList"
    CONST_BASE_MODEL_ID = "baseModelId"
    CONST_MEMBER_MODELS = "memberModels"
    CONST_MODEL_GROUP_VERSION_HISTORY_ID = "modelGroupVersionHistoryId"
    CONST_MODEL_GROUP_VERSION_HISTORY_NAME = "modelGroupVersionHistoryName"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_LIFECYCLE_DETAILS = "lifecycleDetails"
    CONST_TIME_CREATED = "timeCreated"
    CONST_TIME_UPDATED = "timeUpdated"
    CONST_CREATED_BY = "createdBy"
    CONST_VERSION_LABEL = "versionLabel"
    CONST_VERSION_ID = "versionId"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_PROJECT_ID: "project_id",
        CONST_DISPLAY_NAME: "display_name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_LIFECYCLE_DETAILS: "lifecycle_details",
        CONST_TIME_CREATED: "time_created",
        CONST_TIME_UPDATED: "time_updated",
        CONST_CREATED_BY: "created_by",
        CONST_MODEL_GROUP_VERSION_HISTORY_ID: "model_group_version_history_id",
        CONST_MODEL_GROUP_VERSION_HISTORY_NAME: "model_group_version_history_name",
        CONST_VERSION_LABEL: "version_label",
        CONST_VERSION_ID: "version_id",
    }

    def __init__(self, spec=None, **kwargs):
        """Initializes datascience model group.

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
            - display_name: str
            - description: str
            - defined_tags: Dict[str, Dict[str, object]]
            - freeform_tags: Dict[str, str]
            - custom_metadata_list: Union[ModelCustomMetadata, Dict]
            - base_model_id: str
            - member_models: List[Dict[str, str]]
            - model_group_version_history_id: str
            - version_label: str
        """
        super().__init__(spec, **kwargs)
        self.dsc_model_group = OCIDataScienceModelGroup()

    @property
    def kind(self) -> str:
        """The kind of the model group as showing in a YAML."""
        return MODEL_GROUP_KIND

    @property
    def id(self) -> str:
        """The model group OCID."""
        return self.get_spec(self.CONST_ID)

    @property
    def lifecycle_state(self) -> str:
        """The model group lifecycle state."""
        return self.get_spec(self.CONST_LIFECYCLE_STATE)

    @property
    def lifecycle_details(self) -> str:
        """The model group lifecycle details."""
        return self.get_spec(self.CONST_LIFECYCLE_DETAILS)

    @property
    def create_type(self) -> str:
        """The model group create type."""
        return self.get_spec(self.CONST_CREATE_TYPE)

    @property
    def model_group_version_history_name(self) -> str:
        """The model group version history name."""
        return self.get_spec(self.CONST_MODEL_GROUP_VERSION_HISTORY_NAME)

    @property
    def version_id(self) -> str:
        """The model group version id."""
        return self.get_spec(self.CONST_VERSION_ID)

    def with_create_type(self, create_type: str) -> "DataScienceModelGroup":
        """Sets the create type.

        Parameters
        ----------
        create_type: str
            The create type of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        if create_type not in ALLOWED_CREATE_TYPES:
            raise ValueError(
                f"Invalid create type. Allowed create type are {ALLOWED_CREATE_TYPES}."
            )
        return self.set_spec(self.CONST_CREATE_TYPE, create_type)

    @property
    def compartment_id(self) -> str:
        """The model group compartment id."""
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    def with_compartment_id(self, compartment_id: str) -> "DataScienceModelGroup":
        """Sets the compartment OCID.

        Parameters
        ----------
        compartment_id: str
            The compartment id of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def project_id(self) -> str:
        """The model group project id."""
        return self.get_spec(self.CONST_PROJECT_ID)

    def with_project_id(self, project_id: str) -> "DataScienceModelGroup":
        """Sets the project OCID.

        Parameters
        ----------
        project_id: str
            The project id of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def display_name(self) -> str:
        """The model group display name."""
        return self.get_spec(self.CONST_DISPLAY_NAME)

    def with_display_name(self, display_name: str) -> "DataScienceModelGroup":
        """Sets the display name.

        Parameters
        ----------
        display_name: str
            The display name of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_DISPLAY_NAME, display_name)

    @property
    def description(self) -> str:
        """The model group description."""
        return self.get_spec(self.CONST_DESCRIPTION)

    def with_description(self, description: str) -> "DataScienceModelGroup":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def freeform_tags(self) -> Dict[str, str]:
        """The model group freeform tags."""
        return self.get_spec(self.CONST_FREEFORM_TAG)

    def with_freeform_tags(self, **kwargs) -> "DataScienceModelGroup":
        """Sets the freeform tags.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_FREEFORM_TAG, kwargs)

    @property
    def defined_tags(self) -> Dict[str, Dict[str, object]]:
        """The model group defined tags."""
        return self.get_spec(self.CONST_DEFINED_TAG)

    def with_defined_tags(self, **kwargs) -> "DataScienceModelGroup":
        """Sets the defined tags.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_DEFINED_TAG, kwargs)

    @property
    def custom_metadata_list(self) -> ModelCustomMetadata:
        """The model group custom metadata list."""
        return self.get_spec(self.CONST_CUSTOM_METADATA_LIST)

    def with_custom_metadata_list(
        self, metadata: Union[ModelCustomMetadata, Dict]
    ) -> "DataScienceModelGroup":
        """Sets model group custom metadata.

        Parameters
        ----------
        metadata: Union[ModelCustomMetadata, Dict]
            The custom metadata.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        if metadata and isinstance(metadata, Dict):
            metadata = ModelCustomMetadata.from_dict(metadata)
        return self.set_spec(self.CONST_CUSTOM_METADATA_LIST, metadata)

    @property
    def base_model_id(self) -> str:
        """The model group base model id."""
        return self.get_spec(self.CONST_BASE_MODEL_ID)

    def with_base_model_id(self, base_model_id: str) -> "DataScienceModelGroup":
        """Sets base model id.

        Parameters
        ----------
        base_model_id: str
            The base model id.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_BASE_MODEL_ID, base_model_id)

    @property
    def member_models(self) -> List[Dict[str, str]]:
        """The member models of model group."""
        return self.get_spec(self.CONST_MEMBER_MODELS)

    def with_member_models(
        self, member_models: List[Dict[str, str]]
    ) -> "DataScienceModelGroup":
        """Sets member models to be grouped.

        Parameters
        ----------
        member_models: List[Dict[str, str]]
            The member models to be grouped.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_MEMBER_MODELS, member_models)

    @property
    def model_group_version_history_id(self) -> str:
        """The model group version history id."""
        return self.get_spec(self.CONST_MODEL_GROUP_VERSION_HISTORY_ID)

    def with_model_group_version_history_id(
        self, model_group_version_history_id: str
    ) -> "DataScienceModelGroup":
        """Sets model group version history id.

        Parameters
        ----------
        model_group_version_history_id: str
            The model group version history id.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(
            self.CONST_MODEL_GROUP_VERSION_HISTORY_ID, model_group_version_history_id
        )

    @property
    def version_label(self) -> str:
        """The model group version label."""
        return self.get_spec(self.CONST_VERSION_LABEL)

    def with_version_label(self, version_label: str) -> "DataScienceModelGroup":
        """Sets model group version label.

        Parameters
        ----------
        version_label: str
            The model group version label.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self)
        """
        return self.set_spec(self.CONST_VERSION_LABEL, version_label)

    def create(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "DataScienceModelGroup":
        """Creates the datascience model group.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for model group to be created before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        DataScienceModelGroup
           The instance of DataScienceModelGroup.
        """
        response = self.dsc_model_group.create(
            create_model_group_details=CreateModelGroupDetails(
                **batch_convert_case(self._build_model_group_details(), "snake")
            ),
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def _build_model_group_details(self) -> dict:
        """Builds model group details dict for creating or updating oci model group."""
        custom_metadata_list = [
            CustomMetadata(
                key=custom_metadata.key,
                value=custom_metadata.value,
                description=custom_metadata.description,
                category=custom_metadata.category,
            )
            for custom_metadata in self.custom_metadata_list._to_oci_metadata()
        ]
        member_model_details = [
            MemberModelDetails(**member_model) for member_model in self.member_models
        ]

        if self.base_model_id:
            model_group_details = StackedModelGroupDetails(
                custom_metadata_list=custom_metadata_list,
                base_model_id=self.base_model_id,
            )
        else:
            model_group_details = HomogeneousModelGroupDetails(
                custom_metadata_list=custom_metadata_list
            )

        member_model_entries = MemberModelEntries(
            member_model_details=member_model_details
        )

        build_model_group_details = copy.deepcopy(self._spec)
        # pop out the unrequired specs for building `CreateModelGroupDetails` or `UpdateModelGroupDetails`.
        build_model_group_details.pop(self.CONST_CUSTOM_METADATA_LIST, None)
        build_model_group_details.pop(self.CONST_MEMBER_MODELS, None)
        build_model_group_details.pop(self.CONST_BASE_MODEL_ID, None)
        build_model_group_details.update(
            {
                self.CONST_COMPARTMENT_ID: self.compartment_id or COMPARTMENT_OCID,
                self.CONST_PROJECT_ID: self.project_id or PROJECT_OCID,
                self.CONST_MODEL_GROUP_DETAILS: model_group_details,
                self.CONST_MEMBER_MODEL_ENTRIES: member_model_entries,
            }
        )

        return build_model_group_details

    def _update_from_oci_model(
        self, oci_model_group_instance: Union[ModelGroup, ModelGroupSummary]
    ) -> "DataScienceModelGroup":
        """Updates self spec from oci model group instance.

        Parameters
        ----------
        oci_model_group_instance: Union[ModelGroup, ModelGroupSummary]
            The oci model group instance, could be an instance of oci.data_science.models.ModelGroup
            or oci.data_science.models.ModelGroupSummary.

        Returns
        -------
        DataScienceModelGroup
           The instance of DataScienceModelGroup.
        """
        self.dsc_model_group = oci_model_group_instance
        for key, value in self.attribute_map.items():
            if hasattr(oci_model_group_instance, value):
                self.set_spec(key, getattr(oci_model_group_instance, value))

        model_group_details: ModelGroupDetails = (
            oci_model_group_instance.model_group_details
        )
        custom_metadata_list: List[CustomMetadata] = (
            model_group_details.custom_metadata_list
        )
        model_custom_metadata = ModelCustomMetadata()
        for metadata in custom_metadata_list:
            model_custom_metadata.add(
                key=metadata.key,
                value=metadata.value,
                description=metadata.description,
                category=metadata.category,
            )
        self.set_spec(self.CONST_CUSTOM_METADATA_LIST, model_custom_metadata)

        if hasattr(model_group_details, "base_model_id"):
            self.set_spec(self.CONST_BASE_MODEL_ID, model_group_details.base_model_id)

        # only updates member_models when oci_model_group_instance is an instance of
        # oci.data_science.models.ModelGroup as oci.data_science.models.ModelGroupSummary
        # doesn't have member_model_entries property.
        if isinstance(oci_model_group_instance, ModelGroup):
            member_model_entries: MemberModelEntries = (
                oci_model_group_instance.member_model_entries
            )
            member_model_details: List[MemberModelDetails] = (
                member_model_entries.member_model_details
            )

            self.set_spec(
                self.CONST_MEMBER_MODELS,
                [
                    {
                        "inference_key": member_model_detail.inference_key,
                        "model_id": member_model_detail.model_id,
                    }
                    for member_model_detail in member_model_details
                ],
            )

        return self

    def update(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "DataScienceModelGroup":
        """Updates a datascience model group.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for model group to be updated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        DataScienceModelGroup
            The instance of DataScienceModelGroup.
        """
        update_model_group_details = OCIDataScienceModelGroup(
            **self._build_model_group_details()
        ).to_oci_model(UpdateModelGroupDetails)

        response = self.dsc_model_group.update(
            update_model_group_details=update_model_group_details,
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def activate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "DataScienceModelGroup":
        """Activates a datascience model group.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for model group to be activated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        DataScienceModelGroup
            The instance of DataScienceModelGroup.
        """
        response = self.dsc_model_group.activate(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def deactivate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "DataScienceModelGroup":
        """Deactivates a datascience model group.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for model group to be deactivated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        DataScienceModelGroup
            The instance of DataScienceModelGroup.
        """
        response = self.dsc_model_group.deactivate(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def delete(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "DataScienceModelGroup":
        """Deletes a datascience model group.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for model group to be deleted before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        DataScienceModelGroup
            The instance of DataScienceModelGroup.
        """
        response = self.dsc_model_group.delete(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )
        return self._update_from_oci_model(response)

    def sync(self) -> "DataScienceModelGroup":
        """Updates the model group instance from backend.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self).
        """
        if not self.id:
            raise ValueError(
                "Model group needs to be created before it can be fetched."
            )
        return self._update_from_oci_model(OCIDataScienceModelGroup.from_id(self.id))

    @classmethod
    def list(
        cls,
        status: str = None,
        compartment_id: str = None,
        **kwargs,
    ) -> List["DataScienceModelGroup"]:
        """Lists datascience model groups in a given compartment.

        Parameters
        ----------
        status: (str, optional). Defaults to `None`.
                        The status of model group. Allowed values: `ACTIVE`, `CREATING`, `DELETED`, `DELETING`, `FAILED` and `INACTIVE`.
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering model groups.

        Returns
        -------
        List[DataScienceModelGroup]
            The list of the datascience model groups.
        """
        return [
            cls()._update_from_oci_model(model_group_summary)
            for model_group_summary in OCIDataScienceModelGroup.list(
                status=status,
                compartment_id=compartment_id,
                **kwargs,
            )
        ]

    @classmethod
    def from_id(cls, model_group_id: str) -> "DataScienceModelGroup":
        """Loads the model group instance from ocid.

        Parameters
        ----------
        model_group_id: str
            The ocid of model group.

        Returns
        -------
        DataScienceModelGroup
            The DataScienceModelGroup instance (self).
        """
        oci_model_group = OCIDataScienceModelGroup.from_id(model_group_id)
        return cls()._update_from_oci_model(oci_model_group)

    def to_dict(self) -> Dict:
        """Serializes model group to a dictionary.

        Returns
        -------
        dict
            The model group serialized as a dictionary.
        """
        spec = copy.deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": batch_convert_case(spec, "camel"),
        }

    @classmethod
    def from_dict(cls, config: Dict) -> "DataScienceModelGroup":
        """Loads model group instance from a dictionary of configurations.

        Parameters
        ----------
        config: Dict
            A dictionary of configurations.

        Returns
        -------
        DataScienceModelGroup
            The model group instance.
        """
        return cls(spec=batch_convert_case(copy.deepcopy(config["spec"]), "snake"))
