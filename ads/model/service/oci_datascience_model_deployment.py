#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from functools import wraps
import logging
from typing import Callable, List
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.work_request import DataScienceWorkRequest
from ads.config import PROJECT_OCID
from ads.model.deployment.common.utils import OCIClientManager, State
import oci

from oci.data_science.models import (
    CreateModelDeploymentDetails,
    UpdateModelDeploymentDetails,
)

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
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


class MissingModelDeploymentIdError(Exception):  # pragma: no cover
    pass


class MissingModelDeploymentWorkflowIdError(Exception):  # pragma: no cover
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
        dsc_model_deployment = OCIDataScienceModelDeployment.from_id(self.id)
        if dsc_model_deployment.lifecycle_state == self.LIFECYCLE_STATE_ACTIVE:
            raise Exception(
                f"Model deployment {dsc_model_deployment.id} is already in active state."
            )

        if dsc_model_deployment.lifecycle_state == self.LIFECYCLE_STATE_INACTIVE:
            logger.info(f"Activating model deployment `{self.id}`.")
            response = self.client.activate_model_deployment(
                self.id,
            )

            if wait_for_completion:
                self.workflow_req_id = response.headers.get("opc-work-request-id", None)

                try:
                    DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                        progress_bar_description="Activating model deployment",
                        max_wait_time=max_wait_time, 
                        poll_interval=poll_interval
                    )
                except Exception as e:
                    logger.error(
                        "Error while trying to activate model deployment: " + str(e)
                    )

            return self.sync()
        else:
            raise Exception(
                f"Can't activate model deployment {dsc_model_deployment.id} when it's in {dsc_model_deployment.lifecycle_state} state."
            )

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
        print(f"Model Deployment OCID: {self.id}")

        if wait_for_completion:
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)

            try:
                DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                    progress_bar_description="Creating model deployment",
                    max_wait_time=max_wait_time, 
                    poll_interval=poll_interval
                )
            except Exception as e:
                logger.error("Error while trying to create model deployment: " + str(e))

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
        dsc_model_deployment = OCIDataScienceModelDeployment.from_id(self.id)
        if dsc_model_deployment.lifecycle_state == self.LIFECYCLE_STATE_INACTIVE:
            raise Exception(
                f"Model deployment {dsc_model_deployment.id} is already in inactive state."
            )

        if dsc_model_deployment.lifecycle_state == self.LIFECYCLE_STATE_ACTIVE:
            logger.info(f"Deactivating model deployment `{self.id}`.")
            response = self.client.deactivate_model_deployment(
                self.id,
            )

            if wait_for_completion:
                self.workflow_req_id = response.headers.get("opc-work-request-id", None)

                try:
                    DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                        progress_bar_description="Deactivating model deployment",
                        max_wait_time=max_wait_time, 
                        poll_interval=poll_interval
                    )
                except Exception as e:
                    logger.error(
                        "Error while trying to deactivate model deployment: " + str(e)
                    )

            return self.sync()
        else:
            raise Exception(
                f"Can't deactivate model deployment {dsc_model_deployment.id} when it's in {dsc_model_deployment.lifecycle_state} state."
            )

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
        dsc_model_deployment = OCIDataScienceModelDeployment.from_id(self.id)
        if dsc_model_deployment.lifecycle_state in [
            self.LIFECYCLE_STATE_DELETED,
            self.LIFECYCLE_STATE_DELETING,
        ]:
            raise Exception(
                f"Model deployment {dsc_model_deployment.id} is either deleted or being deleted."
            )
        if dsc_model_deployment.lifecycle_state not in [
            self.LIFECYCLE_STATE_ACTIVE,
            self.LIFECYCLE_STATE_FAILED,
            self.LIFECYCLE_STATE_INACTIVE,
        ]:
            raise Exception(
                f"Can't delete model deployment {dsc_model_deployment.id} when it's in {dsc_model_deployment.lifecycle_state} state."
            )
        logger.info(f"Deleting model deployment `{self.id}`.")
        response = self.client.delete_model_deployment(
            self.id,
        )

        if wait_for_completion:
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)

            try:
                DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                    progress_bar_description="Deleting model deployment",
                    max_wait_time=max_wait_time, 
                    poll_interval=poll_interval
                )
            except Exception as e:
                logger.error("Error while trying to delete model deployment: " + str(e))

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
            logger.error("Error while trying to update model deployment: " + str(e))

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
