#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from importlib import metadata

import huggingface_hub
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.errors import AquaResourceAccessError, AquaRuntimeError
from ads.aqua.common.utils import (
    get_huggingface_login_timeout,
    known_realm,
)
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.utils import ui_compatability_check


class ADSVersionHandler(AquaAPIhandler):
    """The handler to get the current version of the ADS."""

    @handle_exceptions
    def get(self):
        self.finish({"data": metadata.version("oracle_ads")})


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
        if ui_compatability_check():
            return self.finish({"status": "ok"})
        elif known_realm():
            return self.finish({"status": "compatible"})
        else:
            raise AquaResourceAccessError(
                "The AI Quick actions extension is not compatible in the given region."
            )


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
]
