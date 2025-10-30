#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from functools import wraps
from typing import Callable

import oci

from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.work_request import DataScienceWorkRequest
from ads.model.deployment.common.utils import OCIClientManager, State

try:
    from oci.data_science.models import CreateModelGroupDetails, UpdateModelGroupDetails
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "The oci model group module was not found. Please run `pip install oci` "
        "to install the latest oci sdk."
    ) from err

logger = logging.getLogger(__name__)

DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
ALLOWED_STATUS = [
    State.ACTIVE.name,
    State.CREATING.name,
    State.DELETED.name,
    State.DELETING.name,
    State.FAILED.name,
    State.INACTIVE.name,
]
MODEL_GROUP_NEEDS_TO_BE_CREATED = (
    "Missing model group id. Model group needs to be created before it can be accessed."
)


def check_for_model_group_id(msg: str = MODEL_GROUP_NEEDS_TO_BE_CREATED):
    """The decorator helping to check if the ID attribute sepcified for a datascience model group.

    Parameters
    ----------
    msg: str
        The message that will be thrown.

    Raises
    ------
    MissingModelGroupIdError
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
                raise MissingModelGroupIdError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class MissingModelGroupIdError(Exception):  # pragma: no cover
    pass


class OCIDataScienceModelGroup(
    OCIDataScienceMixin,
    oci.data_science.models.ModelGroup,
):
    """Represents an OCI Data Science Model Group.
    This class contains all attributes of the `oci.data_science.models.ModelGroup`.
    The main purpose of this class is to link the `oci.data_science.models.ModelGroup`
    and the related client methods.
    Linking the `ModelGroup` (payload) to Create/Update/Delete/Activate/Deactivate methods.

    The `OCIDataScienceModelGroup` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model_group = OCIDataScienceModelGroup(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model_group = OCIDataScienceModelGroup(**payload)

    Methods
    -------
    activate(self, ...) -> "OCIDataScienceModelGroup":
        Activates datascience model group.
    create(self, ...) -> "OCIDataScienceModelGroup"
        Creates datascience model group.
    deactivate(self, ...) -> "OCIDataScienceModelGroup":
        Deactivates datascience model group.
    delete(self, ...) -> "OCIDataScienceModelGroup":
        Deletes datascience model group.
    update(self, ...) -> "OCIDataScienceModelGroup":
        Updates datascience model group.
    list(self, ...) -> list[oci.data_science.models.ModelGroupSummary]:
        List oci.data_science.models.ModelGroupSummary instances within given compartment.
    from_id(cls, model_group: str) -> "OCIDataScienceModelGroup":
        Gets model group by OCID.

    Examples
    --------
    >>> oci_model_group = OCIDataScienceModelGroup.from_id(<model_group_ocid>)
    >>> oci_model_group.deactivate()
    >>> oci_model_group.activate(wait_for_completion=False)
    >>> oci_model_group.description = "A brand new description"
    ... oci_model_group.update()
    >>> oci_model_group.sync()
    >>> oci_model_group.list(status="ACTIVE")
    >>> oci_model_group.delete(wait_for_completion=False)
    """

    def __init__(self, config=None, signer=None, client_kwargs=None, **kwargs):
        super().__init__(config, signer, client_kwargs, **kwargs)
        self.workflow_req_id = None

    def create(
        self,
        create_model_group_details: CreateModelGroupDetails,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelGroup":
        """Creates datascience model group.

        Parameters
        ----------
        create_model_group_details: CreateModelGroupDetails
            An instance of CreateModelGroupDetails which consists of all
            necessary parameters to create a data science model group.
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
        OCIDataScienceModelGroup
            The `OCIDataScienceModelGroup` instance (self).
        """
        response = self.client.create_model_group(create_model_group_details)
        self.update_from_oci_model(response.data)
        logger.info(f"Creating model group `{self.id}`.")
        print(f"Model Group OCID: {self.id}")

        if wait_for_completion:
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)

            try:
                DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                    progress_bar_description="Creating model group",
                    max_wait_time=max_wait_time,
                    poll_interval=poll_interval,
                )
            except Exception as e:
                logger.error("Error while trying to create model group: " + str(e))

        return self.sync()

    @check_for_model_group_id(
        msg="Model group needs to be created before it can be activated or deactivated.."
    )
    def activate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelGroup":
        """Activates datascience model group.

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
        OCIDataScienceModelGroup
            The `OCIDataScienceModelGroup` instance (self).
        """
        dsc_model_group = OCIDataScienceModelGroup.from_id(self.id)
        if dsc_model_group.lifecycle_state == self.LIFECYCLE_STATE_ACTIVE:
            raise Exception(
                f"Model group {dsc_model_group.id} is already in active state."
            )

        if dsc_model_group.lifecycle_state == self.LIFECYCLE_STATE_INACTIVE:
            logger.info(f"Activating model group `{self.id}`.")
            response = self.client.activate_model_group(
                self.id,
            )

            if wait_for_completion:
                self.workflow_req_id = response.headers.get("opc-work-request-id", None)

                try:
                    DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                        progress_bar_description="Activating model group",
                        max_wait_time=max_wait_time,
                        poll_interval=poll_interval,
                    )
                except Exception as e:
                    logger.error(
                        "Error while trying to activate model group: " + str(e)
                    )

            return self.sync()
        else:
            raise Exception(
                f"Can't activate model group {dsc_model_group.id} when it's in {dsc_model_group.lifecycle_state} state."
            )

    @check_for_model_group_id(
        msg="Model group needs to be created before it can be activated or deactivated.."
    )
    def deactivate(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelGroup":
        """Deactivates datascience model group.

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
        OCIDataScienceModelGroup
            The `OCIDataScienceModelGroup` instance (self).
        """
        dsc_model_group = self.from_id(self.id)
        if dsc_model_group.lifecycle_state == self.LIFECYCLE_STATE_INACTIVE:
            raise Exception(
                f"Model group {dsc_model_group.id} is already in inactive state."
            )

        if dsc_model_group.lifecycle_state == self.LIFECYCLE_STATE_ACTIVE:
            logger.info(f"Deactivating model group `{self.id}`.")
            response = self.client.deactivate_model_group(
                self.id,
            )

            if wait_for_completion:
                self.workflow_req_id = response.headers.get("opc-work-request-id", None)

                try:
                    DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                        progress_bar_description="Deactivating model group",
                        max_wait_time=max_wait_time,
                        poll_interval=poll_interval,
                    )
                except Exception as e:
                    logger.error(
                        "Error while trying to deactivate model group: " + str(e)
                    )

            return self.sync()
        else:
            raise Exception(
                f"Can't deactivate model group {dsc_model_group.id} when it's in {dsc_model_group.lifecycle_state} state."
            )

    @check_for_model_group_id(
        msg="Model group needs to be created before it can be deleted."
    )
    def delete(
        self,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelGroup":
        """Deletes datascience model group.

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
        OCIDataScienceModelGroup
            The `OCIDataScienceModelGroup` instance (self).
        """
        dsc_model_group = self.from_id(self.id)
        if dsc_model_group.lifecycle_state in [
            self.LIFECYCLE_STATE_DELETED,
            self.LIFECYCLE_STATE_DELETING,
        ]:
            raise Exception(
                f"Model group {dsc_model_group.id} is either deleted or being deleted."
            )
        if dsc_model_group.lifecycle_state not in [
            self.LIFECYCLE_STATE_ACTIVE,
            self.LIFECYCLE_STATE_FAILED,
            self.LIFECYCLE_STATE_INACTIVE,
        ]:
            raise Exception(
                f"Can't delete model group {dsc_model_group.id} when it's in {dsc_model_group.lifecycle_state} state."
            )
        logger.info(f"Deleting model group `{self.id}`.")
        response = self.client.delete_model_group(
            self.id,
        )

        if wait_for_completion:
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)

            try:
                DataScienceWorkRequest(self.workflow_req_id).wait_work_request(
                    progress_bar_description="Deleting model group",
                    max_wait_time=max_wait_time,
                    poll_interval=poll_interval,
                )
            except Exception as e:
                logger.error("Error while trying to delete model group: " + str(e))

        return self.sync()

    @check_for_model_group_id(
        msg="Model group needs to be created before it can be updated."
    )
    def update(
        self,
        update_model_group_details: UpdateModelGroupDetails,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "OCIDataScienceModelGroup":
        """Updates datascience model group.

        Parameters
        ----------
        update_model_group_details: UpdateModelGroupDetails
            Details to update model group.
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
        OCIDataScienceModelGroup
            The `OCIDataScienceModelGroup` instance (self).
        """
        if wait_for_completion:
            wait_for_states = [
                self.LIFECYCLE_STATE_ACTIVE,
                self.LIFECYCLE_STATE_FAILED,
            ]
        else:
            wait_for_states = []

        try:
            response = self.client_composite.update_model_group_and_wait_for_state(
                self.id,
                update_model_group_details,
                wait_for_states=wait_for_states,
                waiter_kwargs={
                    "max_interval_seconds": poll_interval,
                    "max_wait_seconds": max_wait_time,
                },
            )
            self.workflow_req_id = response.headers.get("opc-work-request-id", None)
        except Exception as e:
            logger.error("Error while trying to update model group: " + str(e))

        return self.sync()

    @classmethod
    def list(
        cls,
        status: str = None,
        compartment_id: str = None,
        **kwargs,
    ) -> list:
        """Lists the model group associated with current compartment id and status

        Parameters
        ----------
        status : str
            Status of model group. Defaults to None.
            Allowed values: `ACTIVE`, `CREATING`, `DELETED`, `DELETING`, `FAILED` and `INACTIVE`.
        compartment_id : str
            Target compartment to list model groups from.
            Defaults to the compartment set in the environment variable "NB_SESSION_COMPARTMENT_OCID".
            If "NB_SESSION_COMPARTMENT_OCID" is not set, the root compartment ID will be used.
            An ValueError will be raised if root compartment ID cannot be determined.
        kwargs :
            The values are passed to oci.data_science.DataScienceClient.list_model_groups.

        Returns
        -------
        list
            A list of oci.data_science.models.ModelGroupSummary objects.

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

        if status is not None:
            if status not in ALLOWED_STATUS:
                raise ValueError(
                    f"Allowed `status` values are: {', '.join(ALLOWED_STATUS)}."
                )
            kwargs["lifecycle_state"] = status

        # https://oracle-cloud-infrastructure-python-sdk.readthedocs.io/en/latest/api/pagination.html#module-oci.pagination
        return oci.pagination.list_call_get_all_results(
            cls().client.list_model_groups, compartment_id, **kwargs
        ).data

    @classmethod
    def from_id(cls, model_group_id: str) -> "OCIDataScienceModelGroup":
        """Gets datascience model group by OCID.

        Parameters
        ----------
        model_group_id: str
            The OCID of the datascience model group.

        Returns
        -------
        OCIDataScienceModelGroup
            An instance of `OCIDataScienceModelGroup`.
        """
        return super().from_ocid(model_group_id)
