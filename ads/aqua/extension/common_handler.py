#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import os
from importlib import metadata

import huggingface_hub
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.errors import AquaRuntimeError
from ads.aqua.common.utils import (
    get_huggingface_login_timeout,
)
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import read_file
from ads.config import CONDA_BUCKET_NAME, CONDA_BUCKET_NS
from ads.opctl.operator.common.utils import default_signer


class ADSVersionHandler(AquaAPIhandler):
    """The handler to get the current version of the ADS."""

    @handle_exceptions
    def get(self):
        self.finish({"data": metadata.version("oracle_ads")})


class AquaVersionHandler(AquaAPIhandler):
    @handle_exceptions
    def get(self):
        """
        Returns the current and latest deployed version of AQUA

        {
          "installed": {
            "aqua": "0.1.3.0",
            "ads": "2.14.2"
          },
          "latest": {
            "aqua": "0.1.4.0",
            "ads": "2.14.4"
          }
        }

        """

        current_aqua_version_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "version.json"
        )
        current_aqua_version = json.loads(read_file(current_aqua_version_path))
        current_ads_version = {"ads": metadata.version("oracle_ads")}
        current_version = {"installed": {**current_aqua_version, **current_ads_version}}
        try:
            latest_version_artifact_path = ObjectStorageDetails(
                CONDA_BUCKET_NAME,
                CONDA_BUCKET_NS,
                "service_pack/aqua_latest_version.json",
            ).path
            latest_version = json.loads(
                read_file(latest_version_artifact_path, auth=default_signer())
            )
        except Exception:
            latest_version = {"latest": current_version["installed"]}
        response = {**current_version, **latest_version}
        return self.finish(response)


class CompatibilityCheckHandler(AquaAPIhandler):
    """The handler to check if the extension is compatible."""

    @handle_exceptions
    def get(self):
        """This method provides the availability status of Aqua. If ODSC_MODEL_COMPARTMENT_OCID environment variable
        is set, then status `ok` is returned. For regions where Aqua is available but the environment variable is not
        set due to accesses/policies, we return the `compatible` status to indicate that the extension can be enabled
        for the selected notebook session.

        Returns
        -------
        status dict:
            ok or compatible

        Raises
        ------
            AquaResourceAccessError: raised when aqua is not accessible in the given session/region.

        """
        return self.finish({"status": "ok"})


class NetworkStatusHandler(AquaAPIhandler):
    """Handler to check internet connection."""

    @handle_exceptions
    def get(self):
        requests.get("https://huggingface.com", timeout=get_huggingface_login_timeout())
        return self.finish({"status": 200, "message": "success"})


class HFLoginHandler(AquaAPIhandler):
    """Handler to login to HF."""

    @handle_exceptions
    def post(self, *args, **kwargs):
        """Handles post request for the HF login.

        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        token = input_data.get("token")

        if not token:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("token"))

        # Login to HF
        try:
            huggingface_hub.login(token=token, new_session=False)
        except Exception as ex:
            raise AquaRuntimeError(
                reason=str(ex), service_payload={"error": type(ex).__name__}
            ) from ex

        return self.finish({"status": 200, "message": "login successful"})


class HFUserStatusHandler(AquaAPIhandler):
    """Handler to check if user logged in to the HF."""

    @handle_exceptions
    def get(self):
        try:
            HfApi().whoami()
        except LocalTokenNotFoundError as err:
            raise AquaRuntimeError(
                "You are not logged in. Please log in to Hugging Face using the `huggingface-cli login` command."
                "See https://huggingface.co/settings/tokens.",
            ) from err

        return self.finish({"status": 200, "message": "logged in"})


__handlers__ = [
    ("ads_version", ADSVersionHandler),
    ("hello", CompatibilityCheckHandler),
    ("network_status", NetworkStatusHandler),
    ("hf_login", HFLoginHandler),
    ("hf_logged_in", HFUserStatusHandler),
    ("aqua_version", AquaVersionHandler),
]
