#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import collections
import copy
import datetime
import oci
import warnings
import time
from typing import Dict, List, Union, Any

import oci.loggingsearch
from ads.common import auth as authutil
import pandas as pd
from ads.model.serde.model_input import JsonModelInputSERDE
from ads.common.oci_logging import (
    LOG_INTERVAL,
    LOG_RECORDS_LIMIT,
    ConsolidatedLog,
    OCILog,
)
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.jobs.builders.base import Builder
from ads.jobs.builders.infrastructure.utils import get_value
from ads.model.common.utils import _is_json_serializable
from ads.model.deployment.common.utils import send_request
from ads.model.deployment.model_deployment_infrastructure import (
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_REPLICA,
    DEFAULT_SHAPE_NAME,
    DEFAULT_OCPUS,
    DEFAULT_MEMORY_IN_GBS,
    MODEL_DEPLOYMENT_INFRASTRUCTURE_TYPE,
    ModelDeploymentInfrastructure,
)
from ads.model.deployment.model_deployment_runtime import (
    ModelDeploymentCondaRuntime,
    ModelDeploymentContainerRuntime,
    ModelDeploymentMode,
    ModelDeploymentRuntime,
    ModelDeploymentRuntimeType,
    OCIModelDeploymentRuntimeType,
)
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)
from ads.common import utils as ads_utils
from .common import utils
from .common.utils import State
from .model_deployment_properties import ModelDeploymentProperties
from oci.data_science.models import (
    LogDetails,
    CreateModelDeploymentDetails,
    UpdateModelDeploymentDetails,
)

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
DEFAULT_WORKFLOW_STEPS = 6
DELETE_WORKFLOW_STEPS = 2
DEACTIVATE_WORKFLOW_STEPS = 2
DEFAULT_RETRYING_REQUEST_ATTEMPTS = 3

MODEL_DEPLOYMENT_KIND = "deployment"
MODEL_DEPLOYMENT_TYPE = "modelDeployment"
MODEL_DEPLOYMENT_INFERENCE_SERVER_TRITON = "TRITON"

MODEL_DEPLOYMENT_RUNTIMES = {
    ModelDeploymentRuntimeType.CONDA: ModelDeploymentCondaRuntime,
    ModelDeploymentRuntimeType.CONTAINER: ModelDeploymentContainerRuntime,
}


class ModelDeploymentLogType:
    PREDICT = "predict"
    ACCESS = "access"


class LogNotConfiguredError(Exception):  # pragma: no cover
    pass


class ModelDeploymentPredictError(Exception):  # pragma: no cover
    pass


class ModelDeployment(Builder):
    """
    A class used to represent a Model Deployment.

    Attributes
    ----------
    config: (dict)
        Deployment configuration parameters
    properties: (ModelDeploymentProperties)
        ModelDeploymentProperties object
    workflow_state_progress: (str)
        Workflow request id
    workflow_steps: (int)
        The number of steps in the workflow
    dsc_model_deployment: (OCIDataScienceModelDeployment)
        The OCIDataScienceModelDeployment instance.
    state: (State)
        Returns the deployment state of the current Model Deployment object
    created_by: (str)
        The user that creates the model deployment
    lifecycle_state: (str)
        Model deployment lifecycle state
    lifecycle_details: (str)
        Model deployment lifecycle details
    time_created: (datetime)
        The time when the model deployment is created
    display_name: (str)
        Model deployment display name
    description: (str)
        Model deployment description
    freeform_tags: (dict)
        Model deployment freeform tags
    defined_tags: (dict)
        Model deployment defined tags
    runtime: (ModelDeploymentRuntime)
        Model deployment runtime
    infrastructure: (ModelDeploymentInfrastructure)
        Model deployment infrastructure


    Methods
    -------
    deploy(wait_for_completion, **kwargs)
        Deploy the current Model Deployment object
    delete(wait_for_completion, **kwargs)
        Deletes the current Model Deployment object
    update(wait_for_completion, **kwargs)
        Updates a model deployment
    activate(wait_for_completion, max_wait_time, poll_interval)
        Activates a model deployment
    deactivate(wait_for_completion, max_wait_time, poll_interval)
        Deactivates a model deployment
    list(status, compartment_id, project_id, **kwargs)
        List model deployment within given compartment and project.
    with_display_name(display_name)
        Sets model deployment display name
    with_description(description)
        Sets model deployment description
    with_freeform_tags(freeform_tags)
        Sets model deployment freeform tags
    with_defined_tags(defined_tags)
        Sets model deployment defined tags
    with_runtime(self, runtime)
        Sets model deployment runtime
    with_infrastructure(self, infrastructure)
        Sets model deployment infrastructure
    from_dict(obj_dict)
        Deserializes model deployment instance from dict
    from_id(id)
        Loads model deployment instance from ocid
    sync()
        Updates the model deployment instance from backend


    Examples
    --------
    >>> # Build model deployment from builder apis:
    >>> ds_model_deployment = (ModelDeployment()
    ...    .with_display_name("TestModelDeployment")
    ...    .with_description("Testing the test model deployment")
    ...    .with_freeform_tags(tag1="val1", tag2="val2")
    ...    .with_infrastructure(
    ...        (ModelDeploymentInfrastructure()
    ...        .with_project_id(<project_id>)
    ...        .with_compartment_id(<compartment_id>)
    ...        .with_shape_name("VM.Standard.E4.Flex")
    ...        .with_shape_config_details(
    ...            ocpus=1,
    ...            memory_in_gbs=16
    ...        )
    ...        .with_replica(1)
    ...        .with_bandwidth_mbps(10)
    ...        .with_web_concurrency(10)
    ...        .with_access_log(
    ...            log_group_id=<log_group_id>,
    ...            log_id=<log_id>
    ...        )
    ...        .with_predict_log(
    ...            log_group_id=<log_group_id>,
    ...            log_id=<log_id>
    ...        ))
    ...    )
    ...    .with_runtime(
    ...        (ModelDeploymentContainerRuntime()
    ...        .with_image(<image>)
    ...        .with_image_digest(<image_digest>)
    ...        .with_entrypoint(<entrypoint>)
    ...        .with_server_port(<server_port>)
    ...        .with_health_check_port(<health_check_port>)
    ...        .with_env({"key":"value"})
    ...        .with_deployment_mode("HTTPS_ONLY")
    ...        .with_model_uri(<model_uri>)
    ...        .with_bucket_uri(<bucket_uri>)
    ...        .with_auth(<auth>)
    ...        .with_timeout(<time_out>))
    ...    )
    ... )
    >>> ds_model_deployment.deploy()
    >>> ds_model_deployment.status
    >>> ds_model_deployment.with_display_name("new name").update()
    >>> ds_model_deployment.deactivate()
    >>> ds_model_deployment.sync()
    >>> ds_model_deployment.list(status="ACTIVE")
    >>> ds_model_deployment.delete()

    >>> # Build model deployment from yaml
    >>> ds_model_deployment = ModelDeployment.from_yaml(uri=<path_to_yaml>)
    """

    _PREFIX = "datascience_model_deployment"

    CONST_ID = "id"
    CONST_CREATED_BY = "createdBy"
    CONST_DISPLAY_NAME = "displayName"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_MODEL_DEPLOYMENT_URL = "modelDeploymentUrl"
    CONST_INFRASTRUCTURE = "infrastructure"
    CONST_RUNTIME = "runtime"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_LIFECYCLE_DETAILS = "lifecycleDetails"
    CONST_TIME_CREATED = "timeCreated"

    attribute_map = {
        CONST_ID: "id",
        CONST_CREATED_BY: "created_by",
        CONST_DISPLAY_NAME: "display_name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_MODEL_DEPLOYMENT_URL: "model_deployment_url",
        CONST_INFRASTRUCTURE: "infrastructure",
        CONST_RUNTIME: "runtime",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_LIFECYCLE_DETAILS: "lifecycle_details",
        CONST_TIME_CREATED: "time_created",
    }

    initialize_spec_attributes = [
        "display_name",
        "description",
        "freeform_tags",
        "defined_tags",
        "infrastructure",
        "runtime",
    ]
    model_input_serializer = JsonModelInputSERDE()

    def __init__(
        self,
        properties: Union[ModelDeploymentProperties, Dict] = None,
        config: Dict = None,
        model_deployment_id: str = None,
        model_deployment_url: str = "",
        spec: Dict = None,
        **kwargs,
    ):
        """Initializes a ModelDeployment object.

        Parameters
        ----------
        properties: (Union[ModelDeploymentProperties, Dict], optional). Defaults to None.
            Object containing deployment properties.
            The properties can be `None` when `kwargs` are used for specifying properties.
        config: (Dict, optional). Defaults to None.
            ADS auth dictionary for OCI authentication.
            This can be generated by calling `ads.common.auth.api_keys()` or `ads.common.auth.resource_principal()`.
            If this is `None` then the `ads.common.default_signer(client_kwargs)` will be used.
        model_deployment_id: (str, optional). Defaults to None.
            Model deployment OCID.
        model_deployment_url: (str, optional). Defaults to empty string.
            Model deployment url.
        spec: (dict, optional). Defaults to None.
            Model deployment spec.
        kwargs:
            Keyword arguments for initializing `ModelDeploymentProperties` or `ModelDeployment`.
        """

        if spec and properties:
            raise ValueError(
                "You can only pass in either `spec` or `properties` to initialize model deployment instance."
            )

        if config:
            warnings.warn(
                "Parameter `config` was deprecated in 2.8.2 from ModelDeployment constructor and will be removed in 3.0.0. Please use `ads.set_auth()` to config the auth information. "
                "Check: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html"
            )

        if properties:
            warnings.warn(
                "Parameter `properties` was deprecated in 2.8.2 from ModelDeployment constructor and will be removed in 3.0.0. Please use `spec` or the builder pattern to initialize model deployment instance. "
                "Check: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html"
            )

        if model_deployment_url or model_deployment_id:
            warnings.warn(
                "Parameter `model_deployment_url` and `model_deployment_id` were deprecated in 2.8.2 from ModelDeployment constructor and will be removed in 3.0.0. These two fields will be auto-populated from the service side. "
                "Check: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html"
            )

        initialize_spec = {}
        initialize_spec_kwargs = {}
        if spec:
            initialize_spec = spec
            initialize_spec_kwargs = self._extract_spec_kwargs(**kwargs)
        elif not properties and not spec:
            if self.CONST_INFRASTRUCTURE in kwargs or self.CONST_RUNTIME in kwargs:
                initialize_spec_kwargs = self._extract_spec_kwargs(**kwargs)

        super().__init__(spec=initialize_spec, **initialize_spec_kwargs)

        self.properties = (
            properties
            if isinstance(properties, ModelDeploymentProperties)
            else ModelDeploymentProperties(oci_model_deployment=properties, **kwargs)
        )

        self.current_state = (
            State._from_str(self.properties.lifecycle_state)
            if self.properties.lifecycle_state
            else State.UNKNOWN
        )

        self._access_log = None
        self._predict_log = None
        self.dsc_model_deployment = OCIDataScienceModelDeployment()

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            deployment
        """
        return MODEL_DEPLOYMENT_KIND

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML.

        Returns
        -------
        str
            deployment
        """
        return MODEL_DEPLOYMENT_TYPE

    @property
    def model_deployment_id(self) -> str:
        """The model deployment ocid.

        Returns
        -------
        str
            The model deployment ocid.
        """
        return self.get_spec(self.CONST_ID, None)

    @property
    def id(self) -> str:
        """The model deployment ocid.

        Returns
        -------
        str
            The model deployment ocid.
        """
        return self.get_spec(self.CONST_ID, None)

    @property
    def created_by(self) -> str:
        """The user that creates the model deployment.

        Returns
        -------
        str
            The user that creates the model deployment.
        """
        return self.get_spec(self.CONST_CREATED_BY, None)

    @property
    def url(self) -> str:
        """Model deployment url.

        Returns
        -------
        str
            Model deployment url.
        """
        return self.get_spec(self.CONST_MODEL_DEPLOYMENT_URL, None)

    @property
    def lifecycle_state(self) -> str:
        """Model deployment lifecycle state.

        Returns
        -------
        str
            Model deployment lifecycle state.
        """
        return self.get_spec(self.CONST_LIFECYCLE_STATE, None)

    @property
    def lifecycle_details(self) -> str:
        """Model deployment lifecycle details.

        Returns
        -------
        str
            Model deployment lifecycle details.
        """
        return self.get_spec(self.CONST_LIFECYCLE_DETAILS, None)

    @property
    def time_created(self) -> datetime:
        """The time when the model deployment is created.

        Returns
        -------
        datetime
            The time when the model deployment is created.
        """
        return self.get_spec(self.CONST_TIME_CREATED, None)

    @property
    def display_name(self) -> str:
        """Model deployment display name.

        Returns
        -------
        str
            Model deployment display name.
        """
        return self.get_spec(self.CONST_DISPLAY_NAME, None)

    def with_display_name(self, display_name: str) -> "ModelDeployment":
        """Sets the name of model deployment.

        Parameters
        ----------
        display_name: str
            The name of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_DISPLAY_NAME, display_name)

    @property
    def description(self) -> str:
        """Model deployment description.

        Returns
        -------
        str
            Model deployment description.
        """
        return self.get_spec(self.CONST_DESCRIPTION, None)

    def with_description(self, description: str) -> "ModelDeployment":
        """Sets the description of model deployment.

        Parameters
        ----------
        description: str
            The description of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def freeform_tags(self) -> Dict:
        """Model deployment freeform tags.

        Returns
        -------
        Dict
            Model deployment freeform tags.
        """
        return self.get_spec(self.CONST_FREEFORM_TAG, {})

    def with_freeform_tags(self, **kwargs) -> "ModelDeployment":
        """Sets the freeform tags of model deployment.

        Parameters
        ----------
        kwargs
            The freeform tags of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_FREEFORM_TAG, kwargs)

    @property
    def defined_tags(self) -> Dict:
        """Model deployment defined tags.

        Returns
        -------
        Dict
            Model deployment defined tags.
        """
        return self.get_spec(self.CONST_DEFINED_TAG, {})

    def with_defined_tags(self, **kwargs) -> "ModelDeployment":
        """Sets the defined tags of model deployment.

        Parameters
        ----------
        kwargs
            The defined tags of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_DEFINED_TAG, kwargs)

    @property
    def runtime(self) -> "ModelDeploymentRuntime":
        """Model deployment runtime.

        Returns
        -------
        ModelDeploymentRuntime
            Model deployment runtime.
        """
        return self.get_spec(self.CONST_RUNTIME, None)

    def with_runtime(self, runtime: ModelDeploymentRuntime) -> "ModelDeployment":
        """Sets the runtime of model deployment.

        Parameters
        ----------
        runtime: ModelDeploymentRuntime
            The runtime of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_RUNTIME, runtime)

    @property
    def infrastructure(self) -> "ModelDeploymentInfrastructure":
        """Model deployment infrastructure.

        Returns
        -------
        ModelDeploymentInfrastructure
            Model deployment infrastructure.
        """
        return self.get_spec(self.CONST_INFRASTRUCTURE, None)

    def with_infrastructure(
        self, infrastructure: ModelDeploymentInfrastructure
    ) -> "ModelDeployment":
        """Sets the infrastructure of model deployment.

        Parameters
        ----------
        infrastructure: ModelDeploymentInfrastructure
            The infrastructure of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self.set_spec(self.CONST_INFRASTRUCTURE, infrastructure)

    def deploy(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        """Deploys the current ModelDeployment object

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for deployment to be deployed before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        ModelDeployment
           The instance of ModelDeployment.
        """
        create_model_deployment_details = (
            self._build_model_deployment_details()
            if self._spec
            else self.properties.build()
        )

        response = self.dsc_model_deployment.create(
            create_model_deployment_details=create_model_deployment_details,
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
    ):
        """Deletes the ModelDeployment

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for deployment to be deleted before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        response = self.dsc_model_deployment.delete(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def update(
        self,
        properties: Union[ModelDeploymentProperties, dict, None] = None,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        **kwargs,
    ):
        """Updates a model deployment

        You can update `model_deployment_configuration_details` and change `instance_shape` and `model_id`
        when the model deployment is in the ACTIVE lifecycle state.
        The `bandwidth_mbps` or `instance_count` can only be updated while the model deployment is in the `INACTIVE` state.
        Changes to the `bandwidth_mbps` or `instance_count` will take effect the next time
        the `ActivateModelDeployment` action is invoked on the model deployment resource.

        Parameters
        ----------
        properties: ModelDeploymentProperties or dict
            The properties for updating the deployment.
        wait_for_completion: bool
            Flag set for whether to wait for deployment to be updated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).
        kwargs:
            dict

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        if properties:
            warnings.warn(
                "Parameter `properties` is deprecated from ModelDeployment `update()` in 2.8.6 and will be removed in 3.0.0. Please use the builder pattern or kwargs to update model deployment instance. "
                "Check: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html"
            )

        updated_properties = properties
        if not isinstance(properties, ModelDeploymentProperties):
            updated_properties = ModelDeploymentProperties(
                oci_model_deployment=properties, **kwargs
            )

        update_model_deployment_details = (
            updated_properties.to_update_deployment()
            if properties or updated_properties.oci_model_deployment or kwargs
            else self._update_model_deployment_details(**kwargs)
        )

        response = self.dsc_model_deployment.update(
            update_model_deployment_details=update_model_deployment_details,
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def watch(
        self,
        log_type: str = None,
        time_start: datetime = None,
        interval: int = LOG_INTERVAL,
        log_filter: str = None,
    ) -> "ModelDeployment":
        """Streams the access and/or predict logs of model deployment.

        Parameters
        ----------
        log_type: str, optional
            The log type. Can be `access`, `predict` or None.
            Defaults to None.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        interval : int, optional
            The time interval between sending each request to pull logs from OCI logging service.
            Defaults to 3.
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        status = ""
        while not self._stop_condition():
            status = self._check_and_print_status(status)
            time.sleep(interval)

        time_start = time_start or self.time_created
        try:
            count = self.logs(log_type).stream(
                source=self.model_deployment_id,
                interval=interval,
                stop_condition=self._stream_stop_condition,
                time_start=time_start,
                log_filter=log_filter,
            )

            if not count:
                print(
                    "No logs in the last 14 days. Please reset time_start to see older logs."
                )
            return self.sync()
        except KeyboardInterrupt:
            print("Stop watching logs.")
            pass

    def _stop_condition(self):
        """Stops the sync once the model deployment is in a terminal state."""
        return self.state in [State.ACTIVE, State.FAILED, State.DELETED, State.INACTIVE]

    def _stream_stop_condition(self):
        """Stops the stream sync once the model deployment is in a terminal state."""
        return self.state in [State.FAILED, State.DELETED, State.INACTIVE]

    def _check_and_print_status(self, prev_status) -> str:
        """Check and print the next status.

        Parameters
        ----------
        prev_status: str
            The previous model deployment status.

        Returns
        -------
        str:
            The next model deployment status.
        """
        status = self._model_deployment_status_text()
        if status != prev_status:
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - {status}")
        return status

    def _model_deployment_status_text(self) -> str:
        """Formats the status message.

        Returns
        -------
        str:
            The model deployment life status and life cycle details.
        """
        details = f", {self.lifecycle_details}" if self.lifecycle_details else ""
        return f"Model Deployment {self.lifecycle_state}" + details

    @property
    def state(self) -> State:
        """Returns the deployment state of the current Model Deployment object"""
        request_attempts = 0
        while request_attempts < DEFAULT_RETRYING_REQUEST_ATTEMPTS:
            request_attempts += 1
            try:
                self.current_state = State._from_str(self.sync().lifecycle_state)
                break
            except:
                pass
            time.sleep(1)

        return self.current_state

    @property
    def status(self) -> State:
        """Returns the deployment state of the current Model Deployment object"""
        return self.state

    def predict(
        self,
        json_input=None,
        data: Any = None,
        serializer: "ads.model.ModelInputSerializer" = model_input_serializer,
        auto_serialize_data: bool = False,
        model_name: str = None,
        model_version: str = None,
        **kwargs,
    ) -> dict:
        """Returns prediction of input data run against the model deployment endpoint.

        Examples
        --------
        >>> import numpy as np
        >>> from ads.model import ModelInputSerializer
        >>> class MySerializer(ModelInputSerializer):
        ...     def serialize(self, data):
        ...         serialized_data = 1
        ...         return serialized_data
        >>> model_deployment = ModelDeployment.from_id(<model_deployment_id>)
        >>> prediction = model_deployment.predict(
        ...        data=np.array([1, 2, 3]),
        ...        serializer=MySerializer(),
        ...        auto_serialize_data=True,
        ... )['prediction']

        Parameters
        ----------
        json_input: Json serializable
            JSON payload for the prediction.
        data: Any
            Data for the prediction.
        serializer: ads.model.ModelInputSerializer
            Defaults to ads.model.JsonModelInputSerializer.
        auto_serialize_data: bool
            Defaults to False. Indicate whether to auto serialize input data using `serializer`.
            If `auto_serialize_data=False`, `data` required to be bytes or json serializable
            and `json_input` required to be json serializable. If `auto_serialize_data` set
            to True, data will be serialized before sending to model deployment endpoint.
        model_name: str
            Defaults to None. When the `inference_server="triton"`, the name of the model to invoke.
        model_version: str
            Defaults to None. When the `inference_server="triton"`, the version of the model to invoke.
        kwargs:
            content_type: str
                Used to indicate the media type of the resource.
                By default, it will be `application/octet-stream` for bytes input and `application/json` otherwise.
                The content-type header will be set to this value when calling the model deployment endpoint.

        Returns
        -------
        dict:
            Prediction results.

        """
        current_state = self.sync().lifecycle_state
        if current_state != State.ACTIVE.name:
            raise ModelDeploymentPredictError(
                "This model deployment is not in active state, you will not be able to use predict end point. "
                f"Current model deployment state: {current_state} "
            )
        endpoint = f"{self.url}/predict"
        signer = authutil.default_signer()["signer"]
        header = {
            "signer": signer,
            "content_type": kwargs.get("content_type", None),
        }
        header.update(kwargs.pop("headers", {}))

        if data is None and json_input is None:
            raise AttributeError(
                "Neither `data` nor `json_input` are provided. You need to provide one of them."
            )
        if data is not None and json_input is not None:
            raise AttributeError(
                "`data` and `json_input` are both provided. You can only use one of them."
            )

        try:
            if auto_serialize_data:
                data = data or json_input
                serialized_data = serializer.serialize(data=data)
                return send_request(
                    data=serialized_data,
                    endpoint=endpoint,
                    is_json_payload=_is_json_serializable(serialized_data),
                    header=header,
                )

            if json_input is not None:
                if not _is_json_serializable(json_input):
                    raise ValueError(
                        "`json_input` must be json serializable. "
                        "Set `auto_serialize_data` to True, or serialize the provided input data first,"
                        "or using `data` to pass binary data."
                    )
                utils.get_logger().warning(
                    "The `json_input` argument of `predict()` will be deprecated soon. "
                    "Please use `data` argument. "
                )
                data = json_input

            is_json_payload = _is_json_serializable(data)
            if not isinstance(data, bytes) and not is_json_payload:
                raise TypeError(
                    "`data` is not bytes or json serializable. Set `auto_serialize_data` to `True` to serialize the input data."
                )
            if model_name and model_version:
                header["model-name"] = model_name
                header["model-version"] = model_version
            elif bool(model_version) ^ bool(model_name):
                raise ValueError(
                    "`model_name` and `model_version` have to be provided together."
                )
            prediction = send_request(
                data=data,
                endpoint=endpoint,
                is_json_payload=is_json_payload,
                header=header,
            )
            return prediction
        except oci.exceptions.ServiceError as ex:
            # When bandwidth exceeds the allocated value, TooManyRequests error (429) will be raised by oci backend.
            if ex.status == 429:
                bandwidth_mbps = self.infrastructure.bandwidth_mbps or DEFAULT_BANDWIDTH_MBPS
                utils.get_logger().warning(
                    f"Load balancer bandwidth exceeds the allocated {bandwidth_mbps} Mbps."
                    "To estimate the actual bandwidth, use formula: (payload size in KB) * (estimated requests per second) * 8 / 1024."
                    "To resolve the issue, try sizing down the payload, slowing down the request rate or increasing the allocated bandwidth."
                )
            raise

    def activate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "ModelDeployment":
        """Activates a model deployment

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for deployment to be activated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        response = self.dsc_model_deployment.activate(
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
    ) -> "ModelDeployment":
        """Deactivates a model deployment

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for deployment to be deactivated before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        response = self.dsc_model_deployment.deactivate(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

        return self._update_from_oci_model(response)

    def _log_details(self, log_type: str = ModelDeploymentLogType.ACCESS):
        """Gets log details for the provided `log_type`.

        Properties
        ----------
        log_type: (str, optional). Defaults to "access".
            The log type. Can be "access" or "predict".

        Returns
        -------
        oci.datascience_model.CategoryLogDetails
            Category log details of the ModelDeployment.

        Raises
        ------
        AttributeError
            Deployment doesn't have requested log configuration.

        """
        if self.properties.category_log_details and getattr(
            self.properties.category_log_details, log_type
        ):
            return getattr(self.properties.category_log_details, log_type)
        elif self.infrastructure:
            category_log_details = self._build_category_log_details()
            log = category_log_details.get(log_type, None)
            if log and log.get("logId", None) and log.get("logGroupId", None):
                return LogDetails(
                    log_id=log.get("logId"), log_group_id=log.get("logGroupId")
                )

        raise LogNotConfiguredError(
            f"Deployment `{self.model_deployment_id}` "
            f"has no `{log_type}` log configuration."
        )

    @property
    def predict_log(self) -> OCILog:
        """Gets the model deployment predict logs object.

        Returns
        -------
        OCILog
            The OCILog object containing the predict logs.
        """
        if not self._predict_log:
            log_details = self._log_details(log_type=ModelDeploymentLogType.PREDICT)
            compartment_id = (
                self.infrastructure.compartment_id
                if self.infrastructure
                else self.properties.compartment_id
            )
            self._predict_log = OCILog(
                compartment_id=compartment_id or COMPARTMENT_OCID,
                id=log_details.log_id,
                log_group_id=log_details.log_group_id,
                source=self.model_deployment_id,
                annotation=ModelDeploymentLogType.PREDICT,
            )
        return self._predict_log

    @property
    def access_log(self) -> OCILog:
        """Gets the model deployment access logs object.

        Returns
        -------
        OCILog
            The OCILog object containing the access logs.
        """
        if not self._access_log:
            log_details = self._log_details(log_type=ModelDeploymentLogType.ACCESS)
            compartment_id = (
                self.infrastructure.compartment_id
                if self.infrastructure
                else self.properties.compartment_id
            )
            self._access_log = OCILog(
                compartment_id=compartment_id or COMPARTMENT_OCID,
                id=log_details.log_id,
                log_group_id=log_details.log_group_id,
                source=self.model_deployment_id,
                annotation=ModelDeploymentLogType.ACCESS,
            )
        return self._access_log

    def logs(self, log_type: str = None) -> ConsolidatedLog:
        """Gets the access or predict logs.

        Parameters
        ----------
        log_type: (str, optional). Defaults to None.
            The log type. Can be "access", "predict" or None.

        Returns
        -------
        ConsolidatedLog
            The ConsolidatedLog object containing the logs.
        """
        loggers = []
        if not log_type:
            try:
                loggers.append(self.access_log)
            except LogNotConfiguredError:
                pass

            try:
                loggers.append(self.predict_log)
            except LogNotConfiguredError:
                pass

            if not loggers:
                raise LogNotConfiguredError(
                    "Neither `predict` nor `access` log was configured for the model deployment."
                )
        elif log_type == ModelDeploymentLogType.ACCESS:
            loggers = [self.access_log]
        elif log_type == ModelDeploymentLogType.PREDICT:
            loggers = [self.predict_log]
        else:
            raise ValueError(
                "Parameter log_type should be either access, predict or None."
            )

        return ConsolidatedLog(*loggers)

    def show_logs(
        self,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = LOG_RECORDS_LIMIT,
        log_type: str = None,
    ):
        """Shows deployment logs as a pandas dataframe.

        Parameters
        ----------
        time_start: (datetime.datetime, optional). Defaults to None.
            Starting date and time in RFC3339 format for retrieving logs.
            Defaults to None. Logs will be retrieved 14 days from now.
        time_end: (datetime.datetime, optional). Defaults to None.
            Ending date and time in RFC3339 format for retrieving logs.
            Defaults to None. Logs will be retrieved until now.
        limit: (int, optional). Defaults to 100.
            The maximum number of items to return.
        log_type: (str, optional). Defaults to None.
            The log type. Can be "access", "predict" or None.

        Returns
        -------
            A pandas DataFrame containing logs.
        """
        logging = self.logs(log_type=log_type)

        def prepare_log_record(log):
            """Converts a log record to ordered dict"""
            log_content = log.get("logContent", {})
            return collections.OrderedDict(
                [
                    ("type", log_content.get("type").split(".")[-1]),
                    ("id", log_content.get("id")),
                    ("message", log_content.get("data", {}).get("message")),
                    ("time", log_content.get("time")),
                ]
            )

        logs = logging.search(
            source=self.model_deployment_id,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )
        return pd.DataFrame([prepare_log_record(log.data) for log in logs])

    def sync(self) -> "ModelDeployment":
        """Updates the model deployment instance from backend.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        return self._update_from_oci_model(
            OCIDataScienceModelDeployment.from_id(self.model_deployment_id)
        )

    @classmethod
    def list(
        cls,
        status: str = None,
        compartment_id: str = None,
        project_id: str = None,
        **kwargs,
    ) -> List["ModelDeployment"]:
        """Lists the model deployments associated with current compartment id and status

        Parameters
        ----------
        status : str
            Status of deployment. Defaults to None.
            Allowed values: `ACTIVE`, `CREATING`, `DELETED`, `DELETING`, `FAILED`, `INACTIVE` and `UPDATING`.
        compartment_id : str
            Target compartment to list deployments from.
            Defaults to the compartment set in the environment variable "NB_SESSION_COMPARTMENT_OCID".
            If "NB_SESSION_COMPARTMENT_OCID" is not set, the root compartment ID will be used.
            An ValueError will be raised if root compartment ID cannot be determined.
        project_id : str
            Target project to list deployments from.
            Defaults to the project id in the environment variable "PROJECT_OCID".
        kwargs :
            The values are passed to oci.data_science.DataScienceClient.list_model_deployments.

        Returns
        -------
        list
            A list of ModelDeployment objects.
        """
        deployments = OCIDataScienceModelDeployment.list(
            status=status,
            compartment_id=compartment_id,
            project_id=project_id,
            **kwargs,
        )
        return [cls()._update_from_oci_model(deployment) for deployment in deployments]

    @classmethod
    def list_df(
        cls,
        status: str = None,
        compartment_id: str = None,
        project_id: str = None,
    ) -> pd.DataFrame:
        """Returns the model deployments associated with current compartment and status
            as a Dataframe that can be easily visualized

        Parameters
        ----------
        status : str
            Status of deployment. Defaults to None.
            Allowed values: `ACTIVE`, `CREATING`, `DELETED`, `DELETING`, `FAILED`, `INACTIVE` and `UPDATING`.
        compartment_id : str
            Target compartment to list deployments from.
            Defaults to the compartment set in the environment variable "NB_SESSION_COMPARTMENT_OCID".
            If "NB_SESSION_COMPARTMENT_OCID" is not set, the root compartment ID will be used.
            An ValueError will be raised if root compartment ID cannot be determined.
        project_id : str
            Target project to list deployments from.
            Defaults to the project id in the environment variable "PROJECT_OCID".

        Returns
        -------
        DataFrame
            pandas Dataframe containing information about the ModelDeployments
        """
        model_deployments = cls.list(
            status=status, compartment_id=compartment_id, project_id=project_id
        )
        if isinstance(status, str) or status == None:
            status = State._from_str(status)
        display = pd.DataFrame()
        ids, urls, status_list = [], [], []
        for model_deployment in model_deployments:
            state_of_model = State._from_str(model_deployment.lifecycle_state)
            if status == State.UNKNOWN or status.name == state_of_model.name:
                ids.append(model_deployment.model_deployment_id)
                urls.append(model_deployment.url)
                status_list.append(model_deployment.lifecycle_state)
        display["deployment_id"] = ids
        display["deployment_url"] = urls
        display["current_state"] = status_list
        return display

    @classmethod
    def from_id(cls, id: str) -> "ModelDeployment":
        """Loads the model deployment instance from ocid.

        Parameters
        ----------
        id: str
            The ocid of model deployment.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance (self).
        """
        oci_model = OCIDataScienceModelDeployment.from_id(id)
        return cls(properties=oci_model)._update_from_oci_model(oci_model)

    @classmethod
    def from_dict(cls, obj_dict: Dict) -> "ModelDeployment":
        """Loads model deployment instance from a dictionary of configurations.

        Parameters
        ----------
        obj_dict: Dict
            A dictionary of configurations.

        Returns
        -------
        ModelDeployment
            The model deployment instance.
        """
        if not isinstance(obj_dict, dict):
            raise ValueError(
                "The config data for initializing the model deployment is invalid."
            )
        spec = ads_utils.batch_convert_case(
            copy.deepcopy(obj_dict.get("spec")), "camel"
        )

        mappings = {
            cls.CONST_INFRASTRUCTURE: {
                MODEL_DEPLOYMENT_INFRASTRUCTURE_TYPE: ModelDeploymentInfrastructure,
            },
            cls.CONST_RUNTIME: {
                ModelDeploymentRuntimeType.CONDA: ModelDeploymentCondaRuntime,
                ModelDeploymentRuntimeType.CONTAINER: ModelDeploymentContainerRuntime,
            },
        }
        model_deployment = cls()

        for key, value in spec.items():
            if key in mappings:
                mapping = mappings[key]
                child_config = value
                if child_config.get("type") not in mapping:
                    raise NotImplementedError(
                        f"{key.title()} type: {child_config.get('type')} is not supported."
                    )
                model_deployment.set_spec(
                    key, mapping[child_config.get("type")].from_dict(child_config)
                )
            else:
                model_deployment.set_spec(key, value)

        return model_deployment

    def to_dict(self, **kwargs) -> Dict:
        """Serializes model deployment to a dictionary.

        Returns
        -------
        dict
            The model deployment serialized as a dictionary.
        """
        spec = copy.deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": ads_utils.batch_convert_case(spec, "camel"),
        }

    def _update_from_oci_model(self, oci_model_instance) -> "ModelDeployment":
        """Updates model deployment instance from OCIDataScienceModelDeployment.

        Parameters
        ----------
        oci_model_instance: OCIDataScienceModelDeployment
            The OCIDataScienceModelDeployment instance.

        Returns
        -------
        ModelDeployment
            The model deployment instance.
        """
        self.dsc_model_deployment = oci_model_instance
        for key, value in self.attribute_map.items():
            if hasattr(oci_model_instance, value):
                self.set_spec(key, getattr(oci_model_instance, value))

        infrastructure = ModelDeploymentInfrastructure()
        self._extract_from_oci_model(
            infrastructure, oci_model_instance, infrastructure.sub_level_attribute_maps
        )

        model_deployment_configuration_details = getattr(
            oci_model_instance, "model_deployment_configuration_details", None
        )
        environment_configuration_details = getattr(
            model_deployment_configuration_details,
            "environment_configuration_details",
            None,
        )
        runtime = (
            ModelDeploymentContainerRuntime()
            if getattr(
                environment_configuration_details,
                "environment_configuration_type",
                None,
            )
            == OCIModelDeploymentRuntimeType.CONTAINER
            else ModelDeploymentCondaRuntime()
        )

        self._extract_from_oci_model(runtime, oci_model_instance)
        infrastructure.set_spec(
            infrastructure.CONST_WEB_CONCURRENCY,
            runtime.env.get("WEB_CONCURRENCY", None),
        )
        if (
            runtime.env.get("CONTAINER_TYPE", None)
            == MODEL_DEPLOYMENT_INFERENCE_SERVER_TRITON
        ):
            runtime.set_spec(
                runtime.CONST_INFERENCE_SERVER,
                MODEL_DEPLOYMENT_INFERENCE_SERVER_TRITON.lower(),
            )

        self.set_spec(self.CONST_INFRASTRUCTURE, infrastructure)
        self.set_spec(self.CONST_RUNTIME, runtime)

        return self

    @staticmethod
    def _extract_from_oci_model(
        dsc_instance: Union[ModelDeploymentInfrastructure, ModelDeploymentRuntime],
        oci_model_instance: OCIDataScienceModelDeployment,
        sub_level: Dict = {},
    ) -> Union[ModelDeploymentInfrastructure, ModelDeploymentRuntime]:
        """Extract attributes from OCIDataScienceModelDeployment.

        Parameters
        ----------
        dsc_instance: Union[ModelDeploymentInfrastructure, ModelDeploymentRuntime]
            The ModelDeploymentInfrastructure or ModelDeploymentRuntime instance.
        oci_model_instance: OCIDataScienceModelDeployment
            The OCIDataScienceModelDeployment instance.
        sub_level: Dict
            The sub level attribute maps of ModelDeploymentInfrastructure or ModelDeploymentRuntime

        Returns
        -------
        Union[ModelDeploymentInfrastructure, ModelDeploymentRuntime]
            The ModelDeploymentInfrastructure or ModelDeploymentRuntime instance.
        """
        for infra_attr, dsc_attr in dsc_instance.payload_attribute_map.items():
            value = get_value(oci_model_instance, dsc_attr)
            if value:
                if infra_attr not in sub_level:
                    dsc_instance._spec[infra_attr] = value
                else:
                    dsc_instance._spec[infra_attr] = {}
                    for sub_infra_attr, sub_dsc_attr in sub_level[infra_attr].items():
                        sub_value = get_value(value, sub_dsc_attr)
                        if sub_value:
                            dsc_instance._spec[infra_attr][sub_infra_attr] = sub_value
        return dsc_instance

    def _build_model_deployment_details(self) -> CreateModelDeploymentDetails:
        """Builds CreateModelDeploymentDetails from model deployment instance.

        Returns
        -------
        CreateModelDeploymentDetails
            The CreateModelDeploymentDetails instance.
        """
        if not (self.infrastructure and self.runtime):
            raise ValueError(
                "Missing parameter runtime or infrastructure. Try reruning it after parameters are fully configured."
            )

        create_model_deployment_details = {
            self.CONST_DISPLAY_NAME: self.display_name or self._random_display_name(),
            self.CONST_DESCRIPTION: self.description,
            self.CONST_DEFINED_TAG: self.defined_tags,
            self.CONST_FREEFORM_TAG: self.freeform_tags,
            self.runtime.CONST_DEPLOYMENT_MODE: self.runtime.deployment_mode
            or ModelDeploymentMode.HTTPS,
            self.infrastructure.CONST_COMPARTMENT_ID: self.infrastructure.compartment_id
            or COMPARTMENT_OCID,
            self.infrastructure.CONST_PROJECT_ID: self.infrastructure.project_id
            or PROJECT_OCID,
            self.infrastructure.CONST_MODEL_DEPLOYMENT_CONFIG_DETAILS: self._build_model_deployment_configuration_details(),
            self.infrastructure.CONST_CATEGORY_LOG_DETAILS: self._build_category_log_details(),
        }

        return OCIDataScienceModelDeployment(
            **create_model_deployment_details
        ).to_oci_model(CreateModelDeploymentDetails)

    def _update_model_deployment_details(
        self, **kwargs
    ) -> UpdateModelDeploymentDetails:
        """Builds UpdateModelDeploymentDetails from model deployment instance.

        Returns
        -------
        UpdateModelDeploymentDetails
            The UpdateModelDeploymentDetails instance.
        """
        if not (self.infrastructure and self.runtime):
            raise ValueError(
                "Missing parameter runtime or infrastructure. Try reruning it after parameters are fully configured."
            )
        self._update_spec(**kwargs)
        update_model_deployment_details = {
            self.CONST_DISPLAY_NAME: self.display_name,
            self.CONST_DESCRIPTION: self.description,
            self.CONST_DEFINED_TAG: self.defined_tags,
            self.CONST_FREEFORM_TAG: self.freeform_tags,
            self.infrastructure.CONST_MODEL_DEPLOYMENT_CONFIG_DETAILS: self._build_model_deployment_configuration_details(),
            self.infrastructure.CONST_CATEGORY_LOG_DETAILS: self._build_category_log_details(),
        }
        return OCIDataScienceModelDeployment(
            **update_model_deployment_details
        ).to_oci_model(UpdateModelDeploymentDetails)

    def _update_spec(self, **kwargs) -> "ModelDeployment":
        """Updates model deployment specs from kwargs.

        Parameters
        ----------
        kwargs:
            display_name: (str)
                Model deployment display name
            description: (str)
                Model deployment description
            freeform_tags: (dict)
                Model deployment freeform tags
            defined_tags: (dict)
                Model deployment defined tags

            Additional kwargs arguments.
            Can be any attribute that `ads.model.deployment.ModelDeploymentCondaRuntime`, `ads.model.deployment.ModelDeploymentContainerRuntime`
            and `ads.model.deployment.ModelDeploymentInfrastructure` accepts.

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        if not kwargs:
            return self

        converted_specs = ads_utils.batch_convert_case(kwargs, "camel")
        specs = {
            "self": self._spec,
            "runtime": self.runtime._spec,
            "infrastructure": self.infrastructure._spec,
        }
        sub_set = {
            self.infrastructure.CONST_ACCESS_LOG,
            self.infrastructure.CONST_PREDICT_LOG,
            self.infrastructure.CONST_SHAPE_CONFIG_DETAILS,
        }
        for spec_value in specs.values():
            for key in spec_value:
                if key in converted_specs:
                    if key in sub_set:
                        for sub_key in converted_specs[key]:
                            converted_sub_key = ads_utils.snake_to_camel(sub_key)
                            spec_value[key][converted_sub_key] = converted_specs[key][
                                sub_key
                            ]
                    else:
                        spec_value[key] = copy.deepcopy(converted_specs[key])
        self = (
            ModelDeployment(spec=specs["self"])
            .with_runtime(
                MODEL_DEPLOYMENT_RUNTIMES[self.runtime.type](spec=specs["runtime"])
            )
            .with_infrastructure(
                ModelDeploymentInfrastructure(spec=specs["infrastructure"])
            )
        )

        return self

    def _build_model_deployment_configuration_details(self) -> Dict:
        """Builds model deployment configuration details from model deployment instance.

        Returns
        -------
        Dict:
            Dict contains model deployment configuration details.
        """
        infrastructure = self.infrastructure
        runtime = self.runtime

        instance_configuration = {
            infrastructure.CONST_INSTANCE_SHAPE_NAME: infrastructure.shape_name
            or DEFAULT_SHAPE_NAME,
        }

        if instance_configuration[infrastructure.CONST_INSTANCE_SHAPE_NAME].endswith(
            "Flex"
        ):
            instance_configuration[
                infrastructure.CONST_MODEL_DEPLOYMENT_INSTANCE_SHAPE_CONFIG_DETAILS
            ] = {
                infrastructure.CONST_OCPUS: infrastructure.shape_config_details.get(
                    "ocpus", None
                )
                or DEFAULT_OCPUS,
                infrastructure.CONST_MEMORY_IN_GBS: infrastructure.shape_config_details.get(
                    "memory_in_gbs", None
                )
                or infrastructure.shape_config_details.get("memoryInGBs", None)
                or DEFAULT_MEMORY_IN_GBS,
            }

        if infrastructure.subnet_id:
            instance_configuration[
                infrastructure.CONST_SUBNET_ID
            ] = infrastructure.subnet_id

        scaling_policy = {
            infrastructure.CONST_POLICY_TYPE: "FIXED_SIZE",
            infrastructure.CONST_INSTANCE_COUNT: infrastructure.replica
            or DEFAULT_REPLICA,
        }

        if not runtime.model_uri:
            raise ValueError(
                "Missing parameter model uri. Try reruning it after model uri is configured."
            )

        model_id = runtime.model_uri
        if not model_id.startswith("ocid"):
            from ads.model.datascience_model import DataScienceModel

            dsc_model = DataScienceModel(
                name=self.display_name,
                compartment_id=self.infrastructure.compartment_id or COMPARTMENT_OCID,
                project_id=self.infrastructure.project_id or PROJECT_OCID,
                artifact=runtime.model_uri,
            ).create(
                bucket_uri=runtime.bucket_uri,
                auth=runtime.auth,
                region=runtime.region,
                overwrite_existing_artifact=runtime.overwrite_existing_artifact,
                remove_existing_artifact=runtime.remove_existing_artifact,
                timeout=runtime.timeout,
            )
            model_id = dsc_model.id

        model_configuration_details = {
            infrastructure.CONST_BANDWIDTH_MBPS: infrastructure.bandwidth_mbps
            or DEFAULT_BANDWIDTH_MBPS,
            infrastructure.CONST_INSTANCE_CONFIG: instance_configuration,
            runtime.CONST_MODEL_ID: model_id,
            infrastructure.CONST_SCALING_POLICY: scaling_policy,
        }

        if runtime.env:
            if not hasattr(
                oci.data_science.models,
                "ModelDeploymentEnvironmentConfigurationDetails",
            ):
                raise EnvironmentError(
                    "Environment variable hasn't been supported in the current OCI SDK installed."
                )

        environment_variables = runtime.env
        if infrastructure.web_concurrency:
            environment_variables["WEB_CONCURRENCY"] = str(
                infrastructure.web_concurrency
            )
            runtime.set_spec(runtime.CONST_ENV, environment_variables)
        if (
            hasattr(runtime, "inference_server")
            and runtime.inference_server
            and runtime.inference_server.upper()
            == MODEL_DEPLOYMENT_INFERENCE_SERVER_TRITON
        ):
            environment_variables[
                "CONTAINER_TYPE"
            ] = MODEL_DEPLOYMENT_INFERENCE_SERVER_TRITON
            runtime.set_spec(runtime.CONST_ENV, environment_variables)
        environment_configuration_details = {
            runtime.CONST_ENVIRONMENT_CONFIG_TYPE: runtime.environment_config_type,
            runtime.CONST_ENVIRONMENT_VARIABLES: runtime.env,
        }

        if runtime.environment_config_type == OCIModelDeploymentRuntimeType.CONTAINER:
            if not hasattr(
                oci.data_science.models,
                "OcirModelDeploymentEnvironmentConfigurationDetails",
            ):
                raise EnvironmentError(
                    "Container runtime hasn't been supported in the current OCI SDK installed."
                )
            environment_configuration_details["image"] = runtime.image
            environment_configuration_details["imageDigest"] = runtime.image_digest
            environment_configuration_details["cmd"] = runtime.cmd
            environment_configuration_details["entrypoint"] = runtime.entrypoint
            environment_configuration_details["serverPort"] = runtime.server_port
            environment_configuration_details[
                "healthCheckPort"
            ] = runtime.health_check_port

        model_deployment_configuration_details = {
            infrastructure.CONST_DEPLOYMENT_TYPE: "SINGLE_MODEL",
            infrastructure.CONST_MODEL_CONFIG_DETAILS: model_configuration_details,
            runtime.CONST_ENVIRONMENT_CONFIG_DETAILS: environment_configuration_details,
        }

        if runtime.deployment_mode == ModelDeploymentMode.STREAM:
            if not hasattr(oci.data_science.models, "StreamConfigurationDetails"):
                raise EnvironmentError(
                    "Model deployment mode hasn't been supported in the current OCI SDK installed."
                )
            model_deployment_configuration_details[
                infrastructure.CONST_STREAM_CONFIG_DETAILS
            ] = {
                runtime.CONST_INPUT_STREAM_IDS: runtime.input_stream_ids,
                runtime.CONST_OUTPUT_STREAM_IDS: runtime.output_stream_ids,
            }

        return model_deployment_configuration_details

    def _build_category_log_details(self) -> Dict:
        """Builds category log details from model deployment instance.

        Returns
        -------
        Dict:
            Dict contains category log details.
        """
        if self.infrastructure.log_group_id and self.infrastructure.log_id:
            log_group_details = {
                self.infrastructure.CONST_LOG_GROUP_ID: self.infrastructure.log_group_id,
                self.infrastructure.CONST_LOG_ID: self.infrastructure.log_id,
            }
            return {
                self.infrastructure.CONST_ACCESS: log_group_details,
                self.infrastructure.CONST_PREDICT: log_group_details,
            }

        logs = {}
        if (
            self.infrastructure.access_log and 
            self.infrastructure.access_log.get(self.infrastructure.CONST_LOG_GROUP_ID, None)
            and self.infrastructure.access_log.get(self.infrastructure.CONST_LOG_ID, None)
        ):
            logs[self.infrastructure.CONST_ACCESS] = {
                self.infrastructure.CONST_LOG_GROUP_ID: self.infrastructure.access_log.get(
                    "logGroupId", None
                ),
                self.infrastructure.CONST_LOG_ID: self.infrastructure.access_log.get(
                    "logId", None
                ),
            }
        if (
            self.infrastructure.predict_log and 
            self.infrastructure.predict_log.get(self.infrastructure.CONST_LOG_GROUP_ID, None)
            and self.infrastructure.predict_log.get(self.infrastructure.CONST_LOG_ID, None)
        ):
            logs[self.infrastructure.CONST_PREDICT] = {
                self.infrastructure.CONST_LOG_GROUP_ID: self.infrastructure.predict_log.get(
                    "logGroupId", None
                ),
                self.infrastructure.CONST_LOG_ID: self.infrastructure.predict_log.get(
                    "logId", None
                ),
            }

        return logs

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{ads_utils.get_random_name_for_resource()}"

    def _extract_spec_kwargs(self, **kwargs) -> Dict:
        """Extract spec related keyword arguments from kwargs.

        Parameters
        ----------
        kwargs

        Returns
        -------
        Dict:
            Dict contains model deployment spec related keyword arguments.
        """
        spec_kwargs = {}
        for attribute in self.initialize_spec_attributes:
            if attribute in kwargs:
                spec_kwargs[attribute] = kwargs[attribute]
        return spec_kwargs

    def build(self) -> "ModelDeployment":
        """Load default values from the environment for the job infrastructure."""
        build_method = getattr(self.infrastructure, "build", None)
        if build_method and callable(build_method):
            build_method()
        else:
            raise NotImplementedError
        return self
