#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import re
import time
from urllib.parse import urlparse

import pandas as pd
from ads.common import auth, oci_client, utils
from ads.common.oci_mixin import OCIWorkRequestMixin
from ads.common.utils import snake_to_camel
from ads.config import JOB_RUN_COMPARTMENT_OCID, NB_SESSION_COMPARTMENT_OCID
from ads.common.object_storage_details import (
    ObjectStorageDetails,
    InvalidObjectStoragePath,
)
from oci import pagination
from oci.data_labeling_service.models import (
    ObjectStorageSnapshotExportDetails,
    SnapshotDatasetDetails,
    WorkRequest,
)
from oci.util import to_dict

logger = logging.getLogger(__name__)
NUM_PROGRESS_BAR = 8


class DataLabeling(OCIWorkRequestMixin):
    """Class for data labeling service. Integrate the data labeling service APIs.

    Examples
    --------
    >>> import ads
    >>> import pandas
    >>> from ads.data_labeling.data_labeling_service import DataLabeling
    >>> ads.set_auth("api_key")
    >>> dls = DataLabeling()
    >>> dls.list_dataset()
    >>> metadata_path = dls.export(dataset_id="your dataset id",
    ...     path="oci://<bucket_name>@<namespace>/folder")
    >>> df = pd.DataFrame.ads.read_labeled_data(metadata_path)
    """

    def __init__(
        self,
        compartment_id: str = None,
        dls_cp_client_auth: dict = None,
        dls_dp_client_auth: dict = None,
    ) -> None:
        """Initialize a DataLabeling class.

        Parameters
        ----------
        compartment_id : str, optional
            OCID of data labeling datasets' compartment
        dls_cp_client_auth : dict, optional
            Data Labeling control plane client auth. Default is None. The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        dls_dp_client_auth : dict, optional
            Data Labeling data plane client auth. Default is None. The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        None
            Nothing.
        """
        self.compartment_id = compartment_id
        if self.compartment_id is None:
            self.compartment_id = (
                NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID
            )

        if not self.compartment_id:
            raise ValueError("The parameter `compartment_id` is required.")

        self.dls_cp_client_auth = dls_cp_client_auth or auth.default_signer()
        self.dls_dp_client_auth = dls_dp_client_auth or auth.default_signer()

        self.dls_dp_client = oci_client.OCIClientFactory(
            **self.dls_dp_client_auth
        ).data_labeling_dp

        self.dls_cp_client = oci_client.OCIClientFactory(
            **self.dls_cp_client_auth
        ).data_labeling_cp

    def list_dataset(self, **kwargs) -> pd.DataFrame:
        """List all the datasets created from the data labeling service under a given compartment.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keyword arguments will be passed to oci.data_labeling_serviceDataLabelingManagementClient.list_datasets method.

        Returns
        -------
        pandas.DataFrame
            pandas dataframe which contains the dataset information.

        Raises
        ------
        Exception
            If pagination.list_call_get_all_results() fails
        """
        try:
            items = pagination.list_call_get_all_results(
                self.dls_cp_client.list_datasets, self.compartment_id, **kwargs
            ).data
        except Exception as e:
            raise e

        df = pd.DataFrame()
        if items:
            df = pd.concat(
                [
                    pd.DataFrame(to_dict(dataset), index=[i])
                    for i, dataset in enumerate(items)
                ],
                axis=0,
            )
            df = (
                df.reset_index(drop=True).set_index("id").drop(columns="compartment_id")
            )
        df.columns = [
            snake_to_camel(name, capitalized_first_token=True) for name in df.columns
        ]
        return df

    def export(self, dataset_id: str, path: str, include_unlabeled=False) -> str:
        """Export dataset based on the dataset_id and save the jsonl files under the path
        (metadata jsonl file and the records jsonl file) to the object storage path provided by the user
        and return the metadata jsonl path.

        Parameters
        ----------
        dataset_id : str
            The dataset id of which the snapshot will be generated.
        path : str
            The object storage path to store the generated snapshot.
            "oci://<bucket_name>@<namespace>/prefix"
        include_unlabeled: bool, Optional. Defaults to False.
            Whether to include unlabeled records or not.

        Returns
        -------
        str
            oci path of the metadata jsonl file.
        """

        if not re.match(r"oci://*@*", path):
            raise InvalidObjectStoragePath(
                "The parameter `path` is not valid. It must follow the pattern 'oci://<bucket_name>@<namespace>/key'."
            )
        url_parse = urlparse(path)
        bucketname = url_parse.username
        namespace = url_parse.hostname
        if not bucketname:
            raise InvalidObjectStoragePath(
                f"The parameter `path` is not valid. The bucket name ({bucketname}) was not found. It must follow the pattern 'oci://<bucket_name>@<namespace>/key'."
            )
        if not namespace:
            raise InvalidObjectStoragePath(
                f"The parameter `path` is not valid. The name space ({namespace}) was not found. It must follow the pattern 'oci://<bucket_name>@<namespace>/key'."
            )

        prefix = url_parse.path.strip("/")
        self.client = self.dls_cp_client

        if not prefix.endswith("/"):
            prefix = prefix + "/"
        os_snapshot_export_detail = ObjectStorageSnapshotExportDetails(
            export_type="OBJECT_STORAGE",
            namespace=namespace,
            bucket=bucketname,
            prefix=prefix,
        )

        snapshot_detail = SnapshotDatasetDetails(
            are_annotations_included=True,
            are_unannotated_records_included=include_unlabeled,
            export_details=os_snapshot_export_detail,
        )
        try:
            snapshot_detail_response = self.dls_cp_client.snapshot_dataset(
                dataset_id, snapshot_detail
            )
        except Exception as error:
            if dataset_id not in self.list_dataset().index:
                raise ValueError(
                    "The parameter `dataset_id` is invalid. "
                    "Use the `.list_dataset()` method to obtain a list of  all available datasets."
                )
            raise error

        res_work_request = self._wait_for_work_request(snapshot_detail_response)

        metadata = res_work_request.data.resources[1].metadata
        return ObjectStorageDetails(
            metadata["BUCKET"], metadata["NAMESPACE"], metadata["OBJECT"]
        ).path

    def _wait_for_work_request(self, snapshot_detail_response):
        successful_state = WorkRequest.STATUS_SUCCEEDED
        wait_for_states = (
            WorkRequest.STATUS_CANCELED,
            WorkRequest.STATUS_CANCELING,
            WorkRequest.STATUS_FAILED,
        )
        work_request_id = snapshot_detail_response.headers["opc-work-request-id"]
        work_request_log_entires = self.dls_cp_client.list_work_request_logs(
            work_request_id
        ).data.items
        i = 0
        res_work_request = self.dls_cp_client.get_work_request(work_request_id)
        self._num_progress_bar = NUM_PROGRESS_BAR
        with utils.get_progress_bar(self._num_progress_bar) as progress:
            while len(work_request_log_entires) <= self._num_progress_bar - 1:
                new_work_request_log_entires = work_request_log_entires[i:]
                for work_request_log_entry in new_work_request_log_entires:
                    progress.update(work_request_log_entry.message)
                    i += 1
                if (
                    len(work_request_log_entires) == self._num_progress_bar - 1
                    or res_work_request.data.status in wait_for_states
                ):
                    progress.update(work_request_log_entires[-1].message)
                    if res_work_request.data.status != successful_state:
                        raise Exception(work_request_log_entires[-1].message)
                    else:
                        break
                time.sleep(i)
                work_request_log_entires = self.dls_cp_client.list_work_request_logs(
                    work_request_id
                ).data.items
                res_work_request = self.dls_cp_client.get_work_request(work_request_id)
        return res_work_request
