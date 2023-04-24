#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from functools import wraps
import json
import time
import logging
from typing import Callable, List
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common import utils as progress_bar_utils
from ads.config import PROJECT_OCID
from ads.model.deployment.common import utils
from ads.model.deployment.common.utils import OCIClientManager, State
import oci

from oci.data_science.models import (
    CreateModelDeploymentDetails,
    UpdateModelDeploymentDetails,
)

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
DEACTIVATE_WORKFLOW_STEPS = 2
DELETE_WORKFLOW_STEPS = 2
ACTIVATE_WORKFLOW_STEPS = 6
CREATE_WORKFLOW_STEPS = 6
ALLOWED_STATUS = [
    State.ACTIVE.name,
    State.CREATING.name,
    State.DELETED.name,
    State.DELETING.name,
    State.FAILED.name,
    State.INACTIVE.name,
    State.UPDATING.name,
]

logger = logging.getLogger(__name__)

MODEL_DEPLOYMENT_NEEDS_TO_BE_DEPLOYED = "Missing model deployment id. Model deployment needs to be deployed before it can be accessed."


def check_for_model_deployment_id(msg: str = MODEL_DEPLOYMENT_NEEDS_TO_BE_DEPLOYED):
    """The decorator helping to check if the ID attribute sepcified for a datascience model deployment.

    Parameters
    ----------
    msg: str
        The message that will be thrown.

    Raises
    ------
    MissingModelDeploymentIdError
        In case if the ID attribute not specified.

    Examples
    --------
    >>> @check_for_id(msg="Some message.")
    ... def test_function(self, name: str, last_name: str)
    ...     pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.id:
                raise MissingModelDeploymentIdError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class MissingModelDeploymentIdError(Exception):
    pass


class MissingModelDeploymentWorkflowIdError(Exception):
    pass


class OCIDataScienceModelDeployment(
    OCIDataScienceMixin,
    oci.data_science.models.ModelDeployment,
):
    """Represents an OCI Data Science Model Deployment.
    This class contains all attributes of the `oci.data_science.models.ModelDeployment`.
    The main purpose of this class is to link the `oci.data_science.models.ModelDeployment`
    and the related client methods.
    Linking the `ModelDeployment` (payload) to Create/Update/Delete/Activate/Deactivate methods.

    The `OCIDataScienceModelDeployment` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model_deployment = OCIDataScienceModelDeployment(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model_deployment = OCIDataScienceModelDeployment(**payload)

    Methods
    -------
    activate(self, ...) -> "OCIDataScienceModelDeployment":
        Activates datascience model deployment.
    create(self, ...) -> "OCIDataScienceModelDeployment"
        Creates datascience model deployment.
    deactivate(self, ...) -> "OCIDataScienceModelDeployment":
        Deactivates datascience model deployment.
    delete(self, ...) -> "OCIDataScienceModelDeployment":
        Deletes datascience model deployment.
    update(self, ...) -> "OCIDataScienceModelDeployment":
        Updates datascience model deployment.
    list(self, ...) -> list[oci.data_science.models.ModelDeployment]:
        List oci.data_science.models.ModelDeployment instances within given compartment and project.
    from_id(cls, model_deployment_id: str) -> "OCIDataScienceModelDeployment":
        Gets model deployment by OCID.

    Examples
    --------
    >>> oci_model_deployment = OCIDataScienceModelDeployment.from_id(<model_deployment_ocid>)
    >>> oci_model_deployment.deactivate()
    >>> oci_model_deployment.activate(wait_for_completion=False)
    >>> oci_model_deployment.description = "A brand new description"
    ... oci_model_deployment.update()
    >>> oci_model_deployment.sync()
    >>> oci_model_deployment.list(status="ACTIVE")
    >>> oci_model_deployment.delete(wait_for_completion=False)
    """

    def __init__(
        self,
        config: dict = None,
        signer: oci.signer.Signer = None,
        client_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(config, signer, client_kwargs, **kwargs)
        self.workflow_req_id = None

    @check_for_model_deployment_id(
        msg="Model deployment needs to be deployed before it can be activated or deactivated."
    )
    def activate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelDeployment":
        """Activates datascience model deployment.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for process to be completed.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        OCIDataScienceModelDeployment
            The `OCIDataScienceModelDeployment` instance (self).
        """
        logger.info(f"Activating model deployment `{self.id}`.")
        response = self.client.activate_model_deployment(
            self.id,
        )

        if wait_for_completion:

            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
            oci_model_deployment_object = self.client.get_model_deployment(self.id).data
            current_state = State._from_str(oci_model_deployment_object.lifecycle_state)
            model_deployment_id = self.id

            try:
                self._wait_for_progress_completion(
                    State.ACTIVE.name,
                    ACTIVATE_WORKFLOW_STEPS,
                    [State.FAILED.name, State.INACTIVE.name],
                    self.workflow_req_id,
                    current_state,
                    model_deployment_id,
                    max_wait_time,
                    poll_interval,
                )
            except Exception as e:
                logger.error(
                    f"Error while trying to activate model deployment: {self.id}"
                )
                raise e

        return self.sync()

    def create(
        self,
        create_model_deployment_details: CreateModelDeploymentDetails,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelDeployment":
        """Creates datascience model deployment.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for process to be completed.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        OCIDataScienceModelDeployment
            The `OCIDataScienceModelDeployment` instance (self).
        """
        response = self.client.create_model_deployment(create_model_deployment_details)
        self.update_from_oci_model(response.data)
        logger.info(f"Creating model deployment `{self.id}`.")

        if wait_for_completion:

            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
            res_payload = json.loads(str(response.data))
            current_state = State._from_str(res_payload["lifecycle_state"])
            model_deployment_id = self.id

            try:
                self._wait_for_progress_completion(
                    State.ACTIVE.name,
                    CREATE_WORKFLOW_STEPS,
                    [State.FAILED.name, State.INACTIVE.name],
                    self.workflow_req_id,
                    current_state,
                    model_deployment_id,
                    max_wait_time,
                    poll_interval,
                )
            except Exception as e:
                logger.error(
                    f"Error while trying to create model deployment: {self.id}"
                )
                raise e

        return self.sync()

    @check_for_model_deployment_id(
        msg="Model deployment needs to be deployed before it can be activated or deactivated."
    )
    def deactivate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelDeployment":
        """Deactivates datascience model deployment.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for process to be completed.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        OCIDataScienceModelDeployment
            The `OCIDataScienceModelDeployment` instance (self).
        """
        logger.info(f"Deactivating model deployment `{self.id}`.")
        response = self.client.deactivate_model_deployment(
            self.id,
        )

        if wait_for_completion:

            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
            oci_model_deployment_object = self.client.get_model_deployment(self.id).data
            current_state = State._from_str(oci_model_deployment_object.lifecycle_state)
            model_deployment_id = self.id

            try:
                self._wait_for_progress_completion(
                    State.INACTIVE.name,
                    DEACTIVATE_WORKFLOW_STEPS,
                    [State.FAILED.name],
                    self.workflow_req_id,
                    current_state,
                    model_deployment_id,
                    max_wait_time,
                    poll_interval,
                )
            except Exception as e:
                logger.error(
                    f"Error while trying to deactivate model deployment: {self.id}"
                )
                raise e

        return self.sync()

    @check_for_model_deployment_id(
        msg="Model deployment needs to be deployed before it can be deleted."
    )
    def delete(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelDeployment":
        """Deletes datascience model deployment.

        Parameters
        ----------
        wait_for_completion: bool
            Flag set for whether to wait for process to be completed.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        OCIDataScienceModelDeployment
            The `OCIDataScienceModelDeployment` instance (self).
        """
        logger.info(f"Deleting model deployment `{self.id}`.")
        response = self.client.delete_model_deployment(
            self.id,
        )

        if wait_for_completion:

            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
            oci_model_deployment_object = self.client.get_model_deployment(self.id).data
            current_state = State._from_str(oci_model_deployment_object.lifecycle_state)
            model_deployment_id = self.id

            try:
                self._wait_for_progress_completion(
                    State.DELETED.name,
                    DELETE_WORKFLOW_STEPS,
                    [State.FAILED.name, State.INACTIVE.name],
                    self.workflow_req_id,
                    current_state,
                    model_deployment_id,
                    max_wait_time,
                    poll_interval,
                )
            except Exception as e:
                logger.error(
                    f"Error while trying to delete model deployment: {self.id}"
                )
                raise e

        return self.sync()

    @check_for_model_deployment_id(
        msg="Model deployment needs to be deployed before it can be updated."
    )
    def update(
        self,
        update_model_deployment_details: UpdateModelDeploymentDetails,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelDeployment":
        """Updates datascience model deployment.

        Parameters
        ----------
        update_model_deployment_details: UpdateModelDeploymentDetails
            Details to update model deployment.
        wait_for_completion: bool
            Flag set for whether to wait for process to be completed.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        OCIDataScienceModelDeployment
            The `OCIDataScienceModelDeployment` instance (self).
        """
        if wait_for_completion:
            wait_for_states = [
                oci.data_science.models.WorkRequest.STATUS_SUCCEEDED,
                oci.data_science.models.WorkRequest.STATUS_FAILED,
            ]
        else:
            wait_for_states = []

        try:
            response = self.client_composite.update_model_deployment_and_wait_for_state(
                self.id,
                update_model_deployment_details,
                wait_for_states=wait_for_states,
                waiter_kwargs={
                    "max_interval_seconds": poll_interval,
                    "max_wait_seconds": max_wait_time,
                },
            )
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
        except Exception as e:
            logger.error(f"Error while trying to update model deployment: {self.id}")
            raise e

        return self.sync()

    @classmethod
    def list(
        cls,
        status: str = None,
        compartment_id: str = None,
        project_id: str = None,
        **kwargs,
    ) -> list:
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
            A list of oci.data_science.models.ModelDeployment objects.

        Raises
        ------
        ValueError
            If compartment_id is not specified and cannot be determined from the environment.
        """
        compartment_id = compartment_id or OCIClientManager().default_compartment_id()

        if not compartment_id:
            raise ValueError(
                "Unable to determine compartment ID from environment. Specify `compartment_id`."
            )

        project_id = project_id or PROJECT_OCID
        if project_id:
            kwargs["project_id"] = project_id

        if status is not None:
            if status not in ALLOWED_STATUS:
                raise ValueError(
                    f"Allowed `status` values are: {', '.join(ALLOWED_STATUS)}."
                )
            kwargs["lifecycle_state"] = status

        # https://oracle-cloud-infrastructure-python-sdk.readthedocs.io/en/latest/api/pagination.html#module-oci.pagination
        return oci.pagination.list_call_get_all_results(
            cls().client.list_model_deployments, compartment_id, **kwargs
        ).data

    @classmethod
    def from_id(cls, model_deployment_id: str) -> "OCIDataScienceModelDeployment":
        """Gets datascience model deployment by OCID.

        Parameters
        ----------
        model_deployment_id: str
            The OCID of the datascience model deployment.

        Returns
        -------
        OCIDataScienceModelDeployment
            An instance of `OCIDataScienceModelDeployment`.
        """
        return super().from_ocid(model_deployment_id)

    def _wait_for_progress_completion(
        self,
        final_state: str,
        work_flow_step: int,
        disallowed_final_states: List[str],
        work_flow_request_id: str,
        state: State,
        model_deployment_id: str,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        """_wait_for_progress_completion blocks until progress is completed.

        Parameters
        ----------
        final_state: str
            Final state of model deployment aimed to be reached.
        work_flow_step: int
            Number of work flow step of the request.
        disallowed_final_states: list[str]
            List of disallowed final state to be reached.
        work_flow_request_id: str
            The id of work flow request.
        state: State
            The current state of model deployment.
        model_deployment_id: str
            The ocid of model deployment.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).
        """

        start_time = time.time()
        prev_message = ""
        prev_workflow_stage_len = 0
        current_state = state or State.UNKNOWN
        with progress_bar_utils.get_progress_bar(work_flow_step) as progress:
            if max_wait_time > 0 and utils.seconds_since(start_time) >= max_wait_time:
                utils.get_logger().error(
                    f"Max wait time ({max_wait_time} seconds) exceeded."
                )
            while (
                max_wait_time < 0 or utils.seconds_since(start_time) < max_wait_time
            ) and current_state.name.upper() != final_state:
                if current_state.name.upper() in disallowed_final_states:
                    utils.get_logger().info(
                        f"Operation failed due to deployment reaching state {current_state.name.upper()}. Use Deployment ID for further steps."
                    )
                    break

                prev_state = current_state.name
                try:
                    model_deployment_payload = json.loads(
                        str(self.client.get_model_deployment(model_deployment_id).data)
                    )
                    current_state = (
                        State._from_str(model_deployment_payload["lifecycle_state"])
                        if "lifecycle_state" in model_deployment_payload
                        else State.UNKNOWN
                    )
                    workflow_payload = self.client.list_work_request_logs(
                        work_flow_request_id
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
                    if prev_state != current_state.name:
                        utils.get_logger().info(
                            f"Status Update: {current_state.name} in {utils.seconds_since(start_time)} seconds"
                        )
                except Exception as e:
                    # utils.get_logger().warning(
                    #     "Unable to update deployment status. Details: %s", format(
                    #         e)
                    # )
                    pass
                time.sleep(poll_interval)
            progress.update("Done")
