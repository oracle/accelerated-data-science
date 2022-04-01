#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import collections
import datetime
import json
import time
from typing import Dict, List, Union

import oci.loggingsearch
import pandas as pd
import requests
from ads.common.auth import default_signer
from ads.common.oci_client import OCIClientFactory
from ads.common.oci_logging import LOG_INTERVAL, LOG_RECORDS_LIMIT, OCILog

from .common import utils
from .common.utils import OCIClientManager, State
from .model_deployment_properties import ModelDeploymentProperties

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 30
DEFAULT_WORKFLOW_STEPS = 6
DEFAULT_RETRYING_REQUEST_ATTEMPTS = 3


TERMINAL_STATES = [State.ACTIVE, State.FAILED, State.DELETED, State.INACTIVE]


class ModelDeploymentLogType:
    PREDICT = "predict"
    ACCESS = "access"


class ModelDeploymentLog(OCILog):
    """The class representing model deployment logs."""

    def __init__(self, model_deployment_id: str, **kwargs) -> None:
        """Initializes an OCI log model for the model deployment.

        Parameters
        ----------
        model_deployment_id: str
            The OCID of the model deployment.
            This parameter will be used as a source field to filter the log records.
        kwargs: dict
            Keyword arguments for initializing ModelDeploymentLog.
        """
        super().__init__(**kwargs)
        self.model_deployment_id = model_deployment_id

    @staticmethod
    def _print(logs: List[Dict]) -> None:
        """Prints the list of provided logs."""
        for log in logs:
            timestamp = log.get("time", "")
            if timestamp:
                timestamp = timestamp.split(".")[0].replace("T", " ")
            else:
                timestamp = ""
            print(f"{timestamp} - {log.get('message')}")

    def tail(
        self, limit=LOG_RECORDS_LIMIT, time_start: datetime.datetime = None
    ) -> None:
        """Prints the most recent log records.

        Parameters
        ----------
        limit: (int, optional). Defaults to 100.
            Maximum number of records to be returned.
        time_start: (datetime.datetime, optional)
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.

        Returns
        -------
        None
            Nothing
        """
        self._print(
            super().tail(
                source=self.model_deployment_id, limit=limit, time_start=time_start
            )
        )

    def head(
        self, limit=LOG_RECORDS_LIMIT, time_start: datetime.datetime = None
    ) -> None:
        """Prints the preceding log records.

        Parameters
        ----------
        limit: (int, optional). Defaults to 100.
            Maximum number of records to be returned.
        time_start: (datetime.datetime, optional)
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.

        Returns
        -------
        None
            Nothing
        """
        self._print(
            super().head(
                source=self.model_deployment_id, limit=limit, time_start=time_start
            )
        )

    def stream(
        self,
        interval: int = LOG_INTERVAL,
        stop_condition: callable = None,
        time_start: datetime.datetime = None,
    ) -> None:
        """Streams logs to console/terminal until `stop_condition()` returns true.

        Parameters
        ----------
        interval: (int, optional). Defaults to 3 seconds.
            The time interval between sending each request to pull logs from OCI.
        stop_condition: (callable, optional). Defaults to None.
            A function to determine if the streaming should stop.
            The log streaming will stop if the function returns true.
        time_start: datetime.datetime
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.

        Returns
        -------
        None
            Nothing
        """
        super().stream(
            source=self.model_deployment_id,
            interval=interval,
            stop_condition=stop_condition,
            time_start=time_start,
        )


class ModelDeployment:
    """
    A class used to represent a Model Deployment.

    Attributes
    ----------
    config: (dict)
        Deployment configuration parameters
    deployment_properties: (ModelDeploymentProperties)
        ModelDeploymentProperties object
    workflow_state_progress: (str)
        Workflow request id
    workflow_steps: (int)
        The number of steps in the workflow
    url: (str)
        The model deployment url endpoint
    ds_client: (DataScienceClient)
        The data science client used by model deployment
    ds_composite_client: (DataScienceCompositeClient)
        The composite data science client used by the model deployment
    workflow_req_id: (str)
        Workflow request id
    model_deployment_id: (str)
        model deployment id
    state: (State)
        Returns the deployment state of the current Model Deployment object

    Methods
    -------
    deploy(wait_for_completion, **kwargs)
        Deploy the current Model Deployment object
    delete(wait_for_completion, **kwargs)
        Deletes the current Model Deployment object
    update(wait_for_completion, **kwargs)
        Updates a model deployment
    list_workflow_logs()
        Returns a list of the steps involved in deploying a model
    """

    def __init__(
        self,
        properties=None,
        config=None,
        workflow_req_id=None,
        model_deployment_id=None,
        model_deployment_url="",
        **kwargs,
    ):
        """Initializes a ModelDeployment

        Parameters
        ----------
        properties: ModelDeploymentProperties or dict
            Object containing deployment properties.
            properties can be None when kwargs are used for specifying properties.
        config: dict
            ADS auth dictionary for OCI authentication.
            This can be generated by calling ads.common.auth.api_keys() or ads.common.auth.resource_principal().
            If this is None, ads.common.default_signer(client_kwargs) will be used.
        workflow_req_id: str
            Workflow request id. Defaults to ""
        model_deployment_id: str
            Model deployment OCID. Defaults to ""
        model_deployment_url: str
            Model deployment url. Defaults to ""
        kwargs:
            Keyword arguments for initializing ModelDeploymentProperties
        """

        if config is None:
            utils.get_logger().info("Using default configuration.")
            config = default_signer()

        # self.config is ADS auth dictionary for OCI authentication.
        self.config = config

        self.properties = (
            properties
            if isinstance(properties, ModelDeploymentProperties)
            else ModelDeploymentProperties(
                oci_model_deployment=properties, config=self.config, **kwargs
            )
        )

        self.current_state = (
            State._from_str(self.properties.lifecycle_state)
            if self.properties.lifecycle_state
            else State.UNKNOWN
        )
        self.url = (
            model_deployment_url
            if model_deployment_url
            else self.properties.model_deployment_url
        )
        self.model_deployment_id = (
            model_deployment_id if model_deployment_id else self.properties.id
        )

        self.workflow_state_progress = []
        self.workflow_steps = DEFAULT_WORKFLOW_STEPS

        client_manager = OCIClientManager(config)
        self.ds_client = client_manager.ds_client
        self.ds_composite_client = client_manager.ds_composite_client
        self.workflow_req_id = workflow_req_id

        if self.ds_client:
            self.log_search_client = OCIClientFactory(**self.config).create_client(
                oci.loggingsearch.LogSearchClient
            )

    def deploy(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        """deploy deploys the current ModelDeployment object

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for deployment to complete before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 600).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 60).

        Returns
        -------
        ModelDeployment
           The instance of ModelDeployment.
        """
        response = self.ds_composite_client.create_model_deployment_and_wait_for_state(
            self.properties.build()
        )
        self.workflow_req_id = response.headers["opc-work-request-id"]
        res_payload = json.loads(str(response.data))
        self.current_state = State._from_str(res_payload["lifecycle_state"])
        self.model_deployment_id = res_payload["id"]
        self.url = res_payload["model_deployment_url"]
        if wait_for_completion:
            try:
                self._wait_for_activation(max_wait_time, poll_interval)
            except Exception as e:
                utils.get_logger().error(f"Error while trying to deploy: {str(e)}")
                raise e
        return self

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
            Flag set for whether to wait for deployment to complete before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 600).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 60).

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """

        response = self.ds_composite_client.delete_model_deployment_and_wait_for_state(
            self.model_deployment_id
        )
        # response.data from deleting model is None, headers are populated
        self.workflow_req_id = response.headers["opc-work-request-id"]
        oci_model_deployment_object = self.ds_client.get_model_deployment(
            self.model_deployment_id
        ).data
        self.current_state = State._from_str(
            oci_model_deployment_object.lifecycle_state
        )
        if wait_for_completion:
            try:
                self._wait_for_deletion(max_wait_time, poll_interval)
            except Exception as e:
                utils.get_logger().error(f"Error while trying to delete: {str(e)}")
                raise e
        return self

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
            Flag set for whether to wait for deployment to complete before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 60).
        kwargs:
            dict

        Returns
        -------
        ModelDeployment
            The instance of ModelDeployment.
        """
        if not isinstance(properties, ModelDeploymentProperties):
            properties = ModelDeploymentProperties(
                oci_model_deployment=properties, config=self.config, **kwargs
            )

        if wait_for_completion:
            wait_for_states = ["SUCCEEDED", "FAILED"]
        else:
            wait_for_states = []

        try:
            response = (
                self.ds_composite_client.update_model_deployment_and_wait_for_state(
                    self.model_deployment_id,
                    properties.to_update_deployment(),
                    wait_for_states=wait_for_states,
                    waiter_kwargs={
                        "max_interval_seconds": poll_interval,
                        "max_wait_seconds": max_wait_time,
                    },
                )
            )
            if "opc-work-request-id" in response.headers:
                self.workflow_req_id = response.headers["opc-work-request-id"]
            # Refresh the properties when model is active
            if wait_for_completion:
                self.properties = ModelDeploymentProperties(
                    oci_model_deployment=self.ds_client.get_model_deployment(
                        self.model_deployment_id
                    ).data,
                    config=self.config,
                )
        except Exception as e:
            utils.get_logger().error(
                "Updating model deployment failed with error: %s", format(e)
            )
            raise e

        return self

    @property
    def state(self) -> State:
        """Returns the deployment state of the current Model Deployment object"""
        request_attempts = 0
        while request_attempts < DEFAULT_RETRYING_REQUEST_ATTEMPTS:
            request_attempts += 1
            try:
                oci_state = self.ds_client.get_model_deployment(
                    retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
                    model_deployment_id=self.model_deployment_id,
                ).data.lifecycle_state
                self.current_state = State._from_str(oci_state)
                break
            except:
                pass
            time.sleep(1)

        return self.current_state

    @property
    def status(self) -> State:
        """Returns the deployment state of the current Model Deployment object"""
        return self.state

    def list_workflow_logs(self) -> list:
        """Returns a list of the steps involved in deploying a model

        Returns
        -------
        list
            List of dictionaries detailing the status of each step in the deployment process.
        """
        if self.workflow_req_id == "" or self.workflow_req_id == None:
            utils.get_logger().info("Workflow req id not available")
            raise Exception
        return self.ds_client.list_work_request_logs(self.workflow_req_id).data

    def predict(self, json_input: dict) -> dict:
        """Returns prediction of input data run against the model deployment endpoint

        Parameters
        ----------
        json_input: dict
            JSON payload for the prediction.

        Returns
        -------
        dict
            Prediction results.

        """
        endpoint = self.url
        signer = self.config.get("signer")
        response = requests.post(
            f"{endpoint}/predict", json=json_input, auth=signer
        ).json()
        return response

    def _wait_for_deletion(
        self,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        """_wait_for_deletion blocks until deletion is complete

        Parameters
        ----------
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 60).
        """

        start_time = time.time()
        prev_message = ""
        if max_wait_time > 0 and utils.seconds_since(start_time) >= max_wait_time:
            utils.get_logger().error(
                f"Max wait time ({max_wait_time} seconds) exceeded."
            )
        while (
            max_wait_time < 0 or utils.seconds_since(start_time) < max_wait_time
        ) and self.current_state.name.upper() != "DELETED":
            if self.current_state.name.upper() == State.FAILED.name:
                utils.get_logger().info(
                    "Deletion Failed. Use Deployment ID for further steps."
                )
                break
            if self.current_state.name.upper() == State.INACTIVE.name:
                utils.get_logger().info("Deployment Inactive")
                break
            prev_state = self.current_state.name
            try:
                model_deployment_payload = json.loads(
                    str(
                        self.ds_client.get_model_deployment(
                            self.model_deployment_id
                        ).data
                    )
                )
                self.current_state = (
                    State._from_str(model_deployment_payload["lifecycle_state"])
                    if "lifecycle_state" in model_deployment_payload
                    else State.UNKNOWN
                )
                workflow_payload = self.ds_client.list_work_request_logs(
                    self.workflow_req_id
                ).data
                if isinstance(workflow_payload, list) and len(workflow_payload) > 0:
                    if prev_message != workflow_payload[-1].message:
                        prev_message = workflow_payload[-1].message
                if prev_state != self.current_state.name:
                    if "model_deployment_url" in model_deployment_payload:
                        self.url = model_deployment_payload["model_deployment_url"]
                    utils.get_logger().info(
                        f"Status Update: {self.current_state.name} in {utils.seconds_since(start_time)} seconds"
                    )
            except Exception as e:
                # utils.get_logger().warning(
                #     "Unable to update deployment status. Details: %s", format(
                #         e)
                # )
                pass
            time.sleep(poll_interval)

    def _wait_for_activation(
        self,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        """_wait_for_activation blocks deployment until activation is complete

        Parameters
        ----------
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 60).
        """

        start_time = time.time()
        prev_message = ""
        prev_workflow_stage_len = 0
        with utils.get_progress_bar(self.workflow_steps) as progress:
            if max_wait_time > 0 and utils.seconds_since(start_time) >= max_wait_time:
                utils.get_logger().error(f"Error: Max wait time exceeded")
            while (
                max_wait_time < 0 or utils.seconds_since(start_time) < max_wait_time
            ) and self.current_state.name.upper() != "ACTIVE":
                if self.current_state.name.upper() == State.FAILED.name:
                    utils.get_logger().info(
                        "Deployment Failed. Use Deployment ID for further steps."
                    )
                    break
                if self.current_state.name.upper() == State.INACTIVE.name:
                    utils.get_logger().info("Deployment Inactive")
                    break
                prev_state = self.current_state.name

                try:
                    model_deployment_payload = json.loads(
                        str(
                            self.ds_client.get_model_deployment(
                                self.model_deployment_id
                            ).data
                        )
                    )
                    self.current_state = (
                        State._from_str(model_deployment_payload["lifecycle_state"])
                        if "lifecycle_state" in model_deployment_payload
                        else State.UNKNOWN
                    )

                    workflow_payload = self.ds_client.list_work_request_logs(
                        self.workflow_req_id
                    ).data
                    if isinstance(workflow_payload, list) and len(workflow_payload) > 0:
                        if prev_message != workflow_payload[-1].message:
                            for _ in range(
                                len(workflow_payload) - prev_workflow_stage_len
                            ):
                                progress.update(workflow_payload[-1].message)
                            prev_workflow_stage_len = len(workflow_payload)
                            prev_message = workflow_payload[-1].message
                            prev_workflow_stage_len = len(workflow_payload)
                    if prev_state != self.current_state.name:
                        if "model_deployment_url" in model_deployment_payload:
                            self.url = model_deployment_payload["model_deployment_url"]
                        utils.get_logger().info(
                            f"Status Update: {self.current_state.name} in {utils.seconds_since(start_time)} seconds"
                        )
                except Exception as e:
                    # utils.get_logger().warning(
                    #     "Unable to update deployment status. Details: %s", format(
                    #         e)
                    # )
                    pass

                time.sleep(poll_interval)
            progress.update("Done")

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
        if not self.properties.category_log_details or not getattr(
            self.properties.category_log_details, log_type
        ):
            raise AttributeError(
                f"Deployment `{self.model_deployment_id}` "
                f"has no `{log_type}` log configuration."
            )
        return getattr(self.properties.category_log_details, log_type)

    @property
    def predict_log(self) -> ModelDeploymentLog:
        """Gets the model deployment predict logs object.

        Returns
        -------
        ModelDeploymentLog
            The ModelDeploymentLog object containing the predict logs.
        """
        log_details = self._log_details(log_type=ModelDeploymentLogType.PREDICT)
        return ModelDeploymentLog(
            model_deployment_id=self.model_deployment_id,
            compartment_id=self.properties.compartment_id,
            id=log_details.log_id,
            log_group_id=log_details.log_group_id,
        )

    @property
    def access_log(self) -> ModelDeploymentLog:
        """Gets the model deployment predict logs object.

        Returns
        -------
        ModelDeploymentLog
            The ModelDeploymentLog object containing the predict logs.
        """
        log_details = self._log_details(log_type=ModelDeploymentLogType.ACCESS)
        return ModelDeploymentLog(
            model_deployment_id=self.model_deployment_id,
            compartment_id=self.properties.compartment_id,
            id=log_details.log_id,
            log_group_id=log_details.log_group_id,
        )

    def logs(self, log_type: str = ModelDeploymentLogType.ACCESS, **kwargs):
        """Gets the access or predict logs.

        Parameters
        ----------
        log_type: (str, optional). Defaults to "access".
            The log type. Can be "access" or "predict".
        kwargs: dict
            Back compatability arguments.

        Returns
        -------
        ModelDeploymentLog
            The ModelDeploymentLog object containing the logs.
        """
        if log_type == ModelDeploymentLogType.ACCESS:
            return self.access_log
        return self.predict_log

    def show_logs(
        self,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit=LOG_RECORDS_LIMIT,
        log_type=ModelDeploymentLogType.ACCESS,
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
        log_type: (str, optional). Defaults to "access".
            The log type. Can be "access" or "predict".

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
