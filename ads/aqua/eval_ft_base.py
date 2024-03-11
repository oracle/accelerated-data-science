#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains base class for EvaluationApp and FineTuningApp."""

from typing import Union

import oci

from ads import set_auth
from ads.aqua import AquaApp
from ads.aqua.utils import logger, query_resource
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.common.utils import extract_region
from ads.config import OCI_ODSC_SERVICE_ENDPOINT, OCI_RESOURCE_PRINCIPAL_VERSION
from ads.model import DataScienceModel


class AquaEvalFTApp(AquaApp):
    """Base Aqua App to contain common components for EvaluationApp and FineTuningApp."""

    def _fetch_jobrun_from_model(
        self,
        model: Union[oci.resource_search.models.ResourceSummary, DataScienceModel],
        use_rqs: bool = True,
        jobrun_id: str = None,
    ) -> Union[
        oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
    ]:
        """Extracts job run id from metadata, and gets related job run information."""
        jobrun_id = jobrun_id or self._get_attribute_from_model_metadata(
            model, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID.value
        )

        logger.debug(f"Fetching associated job run: {jobrun_id}")

        try:
            jobrun = (
                query_resource(jobrun_id, return_all=False)
                if use_rqs
                else self.ds_client.get_job_run(jobrun_id).data
            )
        except Exception as e:
            logger.debug(
                f"Failed to retreive job run: {jobrun_id}. " f"DEBUG INFO: {str(e)}"
            )
            jobrun = None

        return jobrun
