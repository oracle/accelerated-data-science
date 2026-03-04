#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import concurrent.futures
from datetime import datetime, timedelta
from threading import Lock

from cachetools import TTLCache
from oci.exceptions import ServiceError
from oci.identity.models import Compartment

from ads.aqua import logger
from ads.aqua.app import AquaApp
from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaResourceAccessError, AquaValueError
from ads.aqua.common.utils import sanitize_response
from ads.aqua.config.container_config import AquaContainerConfig
from ads.aqua.constants import PRIVATE_ENDPOINT_TYPE
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.config import COMPARTMENT_OCID, DATA_SCIENCE_SERVICE_NAME, TENANCY_OCID
from ads.telemetry import telemetry


class AquaUIApp(AquaApp):
    """Contains APIs for supporting Aqua UI.

    Attributes
    ----------

    Methods
    -------
    list_log_groups(self, **kwargs) -> List[Dict]
        Lists all log groups for the specified compartment or tenancy.
    list_logs(self, **kwargs) -> List[Dict]
        Lists the specified log group's log objects.
    list_compartments(self, **kwargs) -> List[Dict]
        Lists the compartments in a specified compartment.
    list_containers(self, **kwargs) -> AquaContainerConfig
        Containers config to be returned to the client.
    """

    _compartments_cache = TTLCache(
        maxsize=10, ttl=timedelta(hours=2), timer=datetime.now
    )
    _cache_lock = Lock()

    @telemetry(entry_point="plugin=ui&action=list_log_groups", name="aqua")
    def list_log_groups(self, **kwargs) -> str:
        """Lists all log groups for the specified compartment or tenancy. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_log_groups <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_log_groups>`_

        Returns
        -------
            str has json representation of oci.logging.models.log_group.LogGroup
        """

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        res = self.logging_client.list_log_groups(
            compartment_id=compartment_id, **kwargs
        ).data
        return sanitize_response(oci_client=self.logging_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_logs", name="aqua")
    def list_logs(self, **kwargs) -> str:
        """Lists the specified log group's log objects. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as log_group_id, log_type
            for `list_logs <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_logs>`_

        Returns
        -------
        str:
            str has json representation of oci.logging.models.log_summary.LogSummary
        """
        log_group_id = kwargs.pop("log_group_id")

        res = self.logging_client.list_logs(log_group_id=log_group_id, **kwargs).data
        return sanitize_response(oci_client=self.logging_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_capacity_reservations", name="aqua")
    def list_capacity_reservations(self, **kwargs) -> list:
        """
        Lists users compute reservations in a specified compartment

        Returns
        -------
            json representation of `oci.core.models.ComputeCapacityReservationSummary`.

        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        logger.info(f"Loading Capacity reservations from compartment: {compartment_id}")

        reservations = self.compute_client.list_compute_capacity_reservations(
            compartment_id=compartment_id, **kwargs
        )
        return sanitize_response(
            oci_client=self.compute_client, response=reservations.data
        )

    @telemetry(entry_point="plugin=ui&action=list_compartments", name="aqua")
    def list_compartments(self) -> str:
        """Lists the compartments in a tenancy specified by TENANCY_OCID env variable. This is a pass through the OCI list_compartments
        API.

        Returns
        -------
        str:
            str has json representation of oci.identity.models.Compartment
        """
        try:
            if not TENANCY_OCID:
                raise AquaValueError(
                    "TENANCY_OCID must be available in environment"
                    " variables to list the sub compartments."
                )

            if TENANCY_OCID in self._compartments_cache:
                logger.info(
                    f"Returning compartments list in {TENANCY_OCID} from cache."
                )
                return self._compartments_cache.get(TENANCY_OCID)

            compartments = []
            # User may not have permissions to list compartment.
            try:
                compartments.extend(
                    self.list_resource(
                        list_func_ref=self.identity_client.list_compartments,
                        compartment_id=TENANCY_OCID,
                        compartment_id_in_subtree=True,
                        access_level="ANY",
                    )
                )
            except ServiceError:
                logger.error(
                    f"ERROR: Unable to list all sub compartment in tenancy {TENANCY_OCID}."
                )
                try:
                    compartments.append(
                        self.list_resource(
                            list_func_ref=self.identity_client.list_compartments,
                            compartment_id=TENANCY_OCID,
                        )
                    )
                except ServiceError:
                    logger.error(
                        f"ERROR: Unable to list all child compartment in tenancy {TENANCY_OCID}."
                    )
            try:
                root_compartment = self.identity_client.get_compartment(
                    TENANCY_OCID
                ).data
                compartments.insert(0, root_compartment)
            except ServiceError:
                logger.error(
                    f"ERROR: Unable to get details of the root compartment {TENANCY_OCID}."
                )
                compartments.insert(
                    0,
                    Compartment(id=TENANCY_OCID, name=" ** Root - Name N/A **"),
                )
            # convert the string of the results flattened as a dict
            res = sanitize_response(
                oci_client=self.identity_client, response=compartments
            )

            # cache compartment results
            self._compartments_cache.__setitem__(key=TENANCY_OCID, value=res)

            return res

        # todo : update this once exception handling is set up
        except ServiceError as se:
            raise se

    def get_default_compartment(self) -> dict:
        """Returns user compartment OCID fetched from environment variables.

        Returns
        -------
        dict:
            The compartment ocid.
        """
        if not COMPARTMENT_OCID:
            logger.error("No compartment id found from environment variables.")
        return {"compartment_id": COMPARTMENT_OCID}

    def clear_compartments_list_cache(self) -> dict:
        """Allows caller to clear compartments list cache
        Returns
        -------
            dict with the key used, and True if cache has the key that needs to be deleted.
        """
        res = {}
        logger.info("Clearing list_compartments cache")
        with self._cache_lock:
            if TENANCY_OCID in self._compartments_cache:
                self._compartments_cache.pop(key=TENANCY_OCID)
                res = {
                    "key": {
                        "tenancy_ocid": TENANCY_OCID,
                    },
                    "cache_deleted": True,
                }
        return res

    @telemetry(entry_point="plugin=ui&action=list_model_version_sets", name="aqua")
    def list_model_version_sets(self, target_tag: str = None, **kwargs) -> str:
        """Lists all model version sets for the specified compartment or tenancy.

        Parameters
        ----------
        target_tag: str
            Required Tag for the targeting model version sets.
        **kwargs
            Addtional arguments, such as `compartment_id`,
            for `list_model_version_sets <https://docs.oracle.com/en-us/iaas/tools/python/2.121.0/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.list_model_version_sets>`_

        Returns
        -------
            str has json representation of `oci.data_science.models.ModelVersionSetSummary`.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        target_resource = (
            "experiments" if target_tag == Tags.AQUA_EVALUATION else "modelversionsets"
        )
        logger.info(f"Loading {target_resource} from compartment: {compartment_id}")

        items = self.list_resource(
            self.ds_client.list_model_version_sets,
            compartment_id=compartment_id,
            **kwargs,
        )

        if target_tag is not None:
            res = []
            for item in items:
                if target_tag in item.freeform_tags:
                    res.append(item)
        else:
            res = items

        return sanitize_response(oci_client=self.ds_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_buckets", name="aqua")
    def list_buckets(self, **kwargs) -> list:
        """Lists all buckets for the specified compartment.

        Parameters
        ----------
        **kwargs
            Addtional arguments, such as `compartment_id`,
            for `list_buckets <https://docs.oracle.com/en-us/iaas/tools/python/2.122.0/api/object_storage/client/oci.object_storage.ObjectStorageClient.html?highlight=list%20bucket#oci.object_storage.ObjectStorageClient.list_buckets>`_

        Returns
        -------
            str has json representation of `oci.object_storage.models.BucketSummary`."""
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        logger.info(f"Loading buckets summary from compartment: {compartment_id}")

        versioned = kwargs.pop("versioned", False)

        os_client = oc.OCIClientFactory(**default_signer()).object_storage
        namespace_name = os_client.get_namespace(compartment_id=compartment_id).data
        logger.info(f"Object Storage namespace is `{namespace_name}`.")

        response = os_client.list_buckets(
            namespace_name=namespace_name,
            compartment_id=compartment_id,
            limit=1000,
            **kwargs,
        ).data

        if response and versioned:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                result = list(
                    filter(None, executor.map(self._is_bucket_versioned, response))
                )
        else:
            result = response

        return sanitize_response(oci_client=os_client, response=result)

    @staticmethod
    def _is_bucket_versioned(response):
        bucket_name = response.name
        namespace = response.namespace
        bucket_uri = f"oci://{bucket_name}@{namespace}"
        if ObjectStorageDetails.from_path(bucket_uri).is_bucket_versioned():
            return response
        else:
            return None

    @telemetry(entry_point="plugin=ui&action=list_job_shapes", name="aqua")
    def list_job_shapes(self, **kwargs) -> list:
        """Lists all available job shapes for the specified compartment.

        Parameters
        ----------
        **kwargs
            Additional arguments, such as `compartment_id`,
            for `list_job_shapes <https://docs.oracle.com/en-us/iaas/tools/python/2.122.0/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.list_job_shapes>`_

        Returns
        -------
            str has json representation of `oci.data_science.models.JobShapeSummary`."""
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        logger.info(f"Loading job shape summary from compartment: {compartment_id}")

        res = self.ds_client.list_job_shapes(
            compartment_id=compartment_id, **kwargs
        ).data
        return sanitize_response(oci_client=self.ds_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_model_deployment_shapes", name="aqua")
    def list_model_deployment_shapes(self, **kwargs) -> list:
        """Lists all available shapes for model deployment in the specified compartment.

        Parameters
        ----------
        **kwargs
            Additional arguments, such as `compartment_id`,
            for `list_model_deployment_shapes <https://docs.oracle.com/en-us/iaas/api/#/en/data-science/20190101/ModelDeploymentShapeSummary/ListModelDeploymentShapes>`_

        Returns
        -------
        str has json representation of `oci.data_science.models.ModelDeploymentShapeSummary`.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        logger.info(
            f"Loading model deployment shape summary from compartment: {compartment_id}"
        )
        res = self.ds_client.list_model_deployment_shapes(
            compartment_id=compartment_id, **kwargs
        ).data
        return sanitize_response(oci_client=self.ds_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_vcn", name="aqua")
    def list_vcn(self, **kwargs) -> list:
        """Lists the virtual cloud networks (VCNs) in the specified compartment.

        Parameters
        ----------
        **kwargs
            Addtional arguments, such as `compartment_id`,
            for `list_vcns <https://docs.oracle.com/iaas/api/#/en/iaas/20160918/Vcn/ListVcns>`_

        Returns
        -------
            json representation of `oci.core.models.Vcn`."""

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        logger.info(f"Loading VCN list from compartment: {compartment_id}")

        # todo: add _vcn_client in init in AquaApp, then add a property vcn_client which does lazy init
        #   of _vcn_client. Do this for all clients in AquaApp
        vcn_client = oc.OCIClientFactory(**default_signer()).virtual_network
        res = vcn_client.list_vcns(compartment_id=compartment_id).data
        return sanitize_response(oci_client=vcn_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_subnets", name="aqua")
    def list_subnets(self, **kwargs) -> list:
        """Lists the subnets in the specified VCN and the specified compartment.

        Parameters
        ----------
        **kwargs
            Addtional arguments, such as `compartment_id`,
            for `list_vcns <https://docs.oracle.com/iaas/api/#/en/iaas/20160918/Subnet/ListSubnets>`_

        Returns
        -------
            json representation of `oci.core.models.Subnet`."""

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        vcn_id = kwargs.pop("vcn_id", None)
        logger.info(
            f"Loading subnet list from compartment: {compartment_id} for VCN: {vcn_id}"
        )

        vcn_client = oc.OCIClientFactory(**default_signer()).virtual_network
        res = vcn_client.list_subnets(compartment_id=compartment_id, vcn_id=vcn_id).data

        return sanitize_response(oci_client=vcn_client, response=res)

    @telemetry(entry_point="plugin=ui&action=list_private_endpoints", name="aqua")
    def list_private_endpoints(self, **kwargs) -> list:
        """Lists the private endpoints in the specified compartment.
        Data seicne private endpoints have two types: `NOTEBOOK_SESSION` and `MODEL_DEPLOYMENT`.
        This api will by default list `MODEL_DEPLOYMENT` type as needed by AQUA model deployment.

        Parameters
        ----------
        **kwargs
            Addtional arguments, such as `compartment_id`,
            for `list_data_science_private_endpoints <https://docs.oracle.com/en-us/iaas/tools/python/latest/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.list_data_science_private_endpoints>`_

        Returns
        -------
            json representation of `oci.data_science.models.DataSciencePrivateEndpointSummary`.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        resource_type = kwargs.pop("resource_type", PRIVATE_ENDPOINT_TYPE)
        logger.info(f"Loading private endpoints from compartment: {compartment_id}")

        res = self.ds_client.list_data_science_private_endpoints(
            compartment_id=compartment_id, data_science_resource_type=resource_type
        ).data

        return sanitize_response(oci_client=self.ds_client, response=res)

    @telemetry(entry_point="plugin=ui&action=get_shape_availability", name="aqua")
    def get_shape_availability(self, **kwargs):
        """
        For a given compartmentId, resource limit name, and scope, returns the number of available resources associated
        with the given limit.
        Parameters
        ----------
        kwargs
            instance_shape: (str).
                The shape of the instance used for deployment.

            **kwargs
            Addtional arguments, such as `compartment_id`,
            for `get_resource_availability <https://docs.oracle.com/iaas/api/#/en/limits/20181025/ResourceAvailability/GetResourceAvailability>`_

        Returns
        -------
        dict:
            available resource count.

        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        instance_shape = kwargs.pop("instance_shape", None)
        limit_name = kwargs.pop("limit_name", None)

        if not instance_shape:
            raise AquaValueError("instance_shape argument is required.")

        limits_client = oc.OCIClientFactory(**default_signer()).limits

        try:
            res = limits_client.get_resource_availability(
                DATA_SCIENCE_SERVICE_NAME, limit_name, compartment_id, **kwargs
            ).data
        except ServiceError as se:
            raise AquaResourceAccessError(
                f"Could not check limits availability for the shape {instance_shape}. Make sure you have the necessary policy to check limits availability.",
                service_payload=se.args[0] if se.args else None,
            ) from None

        available = res.available

        try:
            cards = int(instance_shape.split(".")[-1])
        except Exception:
            cards = 1

        response = {"available_count": available}

        if available < cards:
            raise AquaValueError(
                f"Inadequate resource is available to create the {instance_shape} resource. The number of available "
                f"resource associated with the limit name {limit_name} is {available}.",
                service_payload=response,
            )

        return response

    @telemetry(entry_point="plugin=ui&action=is_bucket_versioned", name="aqua")
    def is_bucket_versioned(self, bucket_uri: str):
        """Check if the given bucket is versioned. Required check for fine-tuned model creation process where the model
        weights are stored.

        Parameters
        ----------
        bucket_uri

        Returns
        -------
            dict:
                is_versioned flag that informs whether it is versioned or not.

        """
        if ObjectStorageDetails.from_path(bucket_uri).is_bucket_versioned():
            is_versioned = True
            message = f"Model artifact bucket {bucket_uri} is versioned."
        else:
            is_versioned = False
            message = f"Model artifact bucket {bucket_uri} is not versioned. Check if the path exists and enable versioning on the bucket to proceed with model creation."

        return {"is_versioned": is_versioned, "message": message}

    @telemetry(entry_point="plugin=ui&action=list_containers", name="aqua")
    def list_containers(self) -> AquaContainerConfig:
        """
        Lists the AQUA containers.

        Returns
        -------
        AquaContainerConfig
            The AQUA containers configurations.
        """
        return AquaContainerConfig.from_service_config(
            service_containers=self.list_service_containers()
        )
