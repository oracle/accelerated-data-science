#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Optional
from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.enums import CustomInferenceContainerTypeFamily
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.aqua.common.utils import get_hf_model_info, is_valid_ocid, list_hf_models
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.model import AquaModelApp
from ads.aqua.model.entities import AquaModelSummary, HFModelSummary


class AquaModelHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    @handle_exceptions
    def get(
        self,
        model_id="",
    ):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/model/files"):
            os_path = self.get_argument("os_path", None)
            model_name = self.get_argument("model_name", None)

            model_format = self.get_argument("model_format")
            if not model_format:
                raise HTTPError(
                    400, Errors.MISSING_REQUIRED_PARAMETER.format("model_format")
                )
            
            model_format = model_format.upper()
            
            if os_path:
                return self.finish(
                    AquaModelApp.get_model_files(os_path, model_format)
                )
            elif model_name:
                return self.finish(
                    AquaModelApp.get_hf_model_files(model_name, model_format)
                )
            else:
                raise HTTPError(
                    400,
                    Errors.MISSING_ONEOF_REQUIRED_PARAMETER.format(
                        "os_path", "model_name"
                    ),
                )
        elif not model_id:
            return self.list()

        return self.read(model_id)

    def read(self, model_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaModelApp().get(model_id))

    @handle_exceptions
    def delete(self, id=""):
        """Handles DELETE request for clearing cache"""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/model/cache"):
            return self.finish(AquaModelApp().clear_model_list_cache())
        elif id:
            return self.finish(AquaModelApp().delete_model(id))
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=None)
        # project_id is no needed.
        project_id = self.get_argument("project_id", default=None)
        model_type = self.get_argument("model_type", default=None)
        return self.finish(
            AquaModelApp().list(
                compartment_id=compartment_id,
                project_id=project_id,
                model_type=model_type,
            )
        )

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles post request for the registering any Aqua model.
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        # required input parameters
        model = input_data.get("model")
        if not model:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model"))
        os_path = input_data.get("os_path")
        if not os_path:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("os_path"))

        inference_container = input_data.get("inference_container")
        finetuning_container = input_data.get("finetuning_container")
        compartment_id = input_data.get("compartment_id")
        project_id = input_data.get("project_id")
        model_file = input_data.get("model_file")
        download_from_hf = (
            str(input_data.get("download_from_hf", "false")).lower() == "true"
        )
        local_dir = input_data.get("local_dir")
        cleanup_model_cache = (
            str(input_data.get("cleanup_model_cache", "false")).lower() == "true"
        )
        inference_container_uri = input_data.get("inference_container_uri")
        allow_patterns = input_data.get("allow_patterns")
        ignore_patterns = input_data.get("ignore_patterns")
        freeform_tags = input_data.get("freeform_tags")
        defined_tags = input_data.get("defined_tags")
        ignore_model_artifact_check = (
            str(input_data.get("ignore_model_artifact_check", "false")).lower()
            == "true"
        )

        return self.finish(
            AquaModelApp().register(
                model=model,
                os_path=os_path,
                download_from_hf=download_from_hf,
                local_dir=local_dir,
                cleanup_model_cache=cleanup_model_cache,
                inference_container=inference_container,
                finetuning_container=finetuning_container,
                compartment_id=compartment_id,
                project_id=project_id,
                model_file=model_file,
                inference_container_uri=inference_container_uri,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
                ignore_model_artifact_check=ignore_model_artifact_check,
            )
        )

    @handle_exceptions
    def put(self, id):
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        inference_container = input_data.get("inference_container")
        inference_container_uri = input_data.get("inference_container_uri")
        inference_containers = AquaModelApp.list_valid_inference_containers()
        inference_containers.extend(CustomInferenceContainerTypeFamily.values())
        if (
            inference_container is not None
            and inference_container not in inference_containers
        ):
            raise HTTPError(
                400, Errors.INVALID_VALUE_OF_PARAMETER.format("inference_container")
            )

        enable_finetuning = input_data.get("enable_finetuning")
        task = input_data.get("task")
        app = AquaModelApp()
        self.finish(
            app.edit_registered_model(
                id,
                inference_container,
                inference_container_uri,
                enable_finetuning,
                task,
            )
        )
        app.clear_model_details_cache(model_id=id)


class AquaModelLicenseHandler(AquaAPIhandler):
    """Handler for Aqua Model license REST APIs."""

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""

        model_id = model_id.split("/")[0]
        return self.finish(AquaModelApp().load_license(model_id))


class AquaHuggingFaceHandler(AquaAPIhandler):
    """Handler for Aqua Hugging Face REST APIs."""

    @staticmethod
    def _find_matching_aqua_model(model_id: str) -> Optional[AquaModelSummary]:
        """
        Finds a matching model in AQUA based on the model ID from Hugging Face.

        Parameters
        ----------
        model_id (str): The Hugging Face model ID to match.

        Returns
        -------
        Optional[AquaModelSummary]
            Returns the matching AquaModelSummary object if found, else None.
        """
        # Convert the Hugging Face model ID to lowercase once
        model_id_lower = model_id.lower()

        aqua_model_app = AquaModelApp()
        model_ocid = aqua_model_app._find_matching_aqua_model(model_id=model_id_lower)
        if model_ocid:
            return aqua_model_app.get(model_ocid, load_model_card=False)

        return None

    @handle_exceptions
    def get(self, *args, **kwargs):  # noqa: ARG002
        """
        Finds a list of matching models from hugging face based on query string provided from users.

        Parameters
        ----------
        query (str): The Hugging Face model name to search for.

        Returns
        -------
        List[str]
            Returns the matching model ids string
        """

        query = self.get_argument("query", default=None)
        if not query:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("query"))
        models = list_hf_models(query)
        return self.finish({"models": models})

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """Handles post request for the HF Models APIs

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

        model_id = input_data.get("model_id")

        if not model_id:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model_id"))

        # Get model info from the HF
        hf_model_info = get_hf_model_info(repo_id=model_id)

        # Check if model is not disabled
        if hf_model_info.disabled:
            raise AquaRuntimeError(
                f"The chosen model '{hf_model_info.id}' is currently disabled and cannot be imported into AQUA. "
                "Please verify the model's status on the Hugging Face Model Hub or select a different model."
            )

        # Commented the validation below to let users to register any model type.
        # # Check pipeline_tag, it should be `text-generation`
        # if not (
        #     hf_model_info.pipeline_tag
        #     and hf_model_info.pipeline_tag.lower() in ModelTask
        # ):
        #     raise AquaRuntimeError(
        #         f"Unsupported pipeline tag for the chosen model: '{hf_model_info.pipeline_tag}'. "
        #         f"AQUA currently supports the following tasks only: {', '.join(ModelTask.values())}. "
        #         "Please select a model with a compatible pipeline tag."
        #     )

        # Check if it is a service/verified model
        aqua_model_info: AquaModelSummary = self._find_matching_aqua_model(
            model_id=hf_model_info.id
        )

        return self.finish(
            HFModelSummary(model_info=hf_model_info, aqua_model_info=aqua_model_info)
        )


class AquaModelTokenizerConfigHandler(AquaAPIhandler):
    def get(self, model_id):
        """
        Handles requests for retrieving the Hugging Face tokenizer configuration of a specified model.
        Expected request format: GET /aqua/models/<model-ocid>/tokenizer

        """

        path_list = urlparse(self.request.path).path.strip("/").split("/")
        # Path should be /aqua/models/ocid1.iad.ahdxxx/tokenizer
        # path_list=['aqua','models','<model-ocid>','tokenizer']
        if (
            len(path_list) == 4
            and is_valid_ocid(path_list[2])
            and path_list[3] == "tokenizer"
        ):
            return self.finish(AquaModelApp().get_hf_tokenizer_config(model_id))

        raise HTTPError(400, f"The request {self.request.path} is invalid.")


__handlers__ = [
    ("model/?([^/]*)", AquaModelHandler),
    ("model/?([^/]*)/license", AquaModelLicenseHandler),
    ("model/?([^/]*)/tokenizer", AquaModelTokenizerConfigHandler),
    ("model/hf/search/?([^/]*)", AquaHuggingFaceHandler),
]
