#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import shutil
import os
import re
import json
import requests
from typing import List, Union, Optional, Dict, Any

from pydantic import ValidationError
from rich.table import Table

from ads.aqua.app import logger
from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import (
    AquaFileNotFoundError,
    AquaRecommendationError,
    AquaValueError,
)
from ads.aqua.common.utils import (
    build_pydantic_error_message,
    get_resource_type,
    load_config,
    load_gpu_shapes_index,
    is_valid_ocid,
)
from ads.aqua.shaperecommend.constants import (
    BITS_AND_BYTES_4BIT,
    BITSANDBYTES,
    SAFETENSORS,
    SHAPE_MAP,
    TEXT_GENERATION,
    TROUBLESHOOT_MSG,
    HUGGINGFACE_CONFIG_URL,
)
from ads.aqua.shaperecommend.estimator import get_estimator
from ads.aqua.shaperecommend.llm_config import LLMConfig
from ads.aqua.shaperecommend.shape_report import (
    ModelConfig,
    RequestRecommend,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.model.datascience_model import DataScienceModel
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)


class HuggingFaceModelFetcher:
    """
    Utility class to fetch model configurations from HuggingFace.
    """

    @classmethod
    def is_huggingface_model_id(cls, model_id: str) -> bool:
        if is_valid_ocid(model_id):
            return False
        hf_pattern = r"^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)?$"
        return bool(re.match(hf_pattern, model_id))

    @classmethod
    def get_hf_token(cls) -> Optional[str]:
        return os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    @classmethod
    def fetch_config_only(cls, model_id: str) -> Dict[str, Any]:
        try:
            config_url = HUGGINGFACE_CONFIG_URL.format(model_id=model_id)
            headers = {}
            token = cls.get_hf_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(config_url, headers=headers, timeout=10)
            if response.status_code == 401:
                raise AquaValueError(
                    f"Model '{model_id}' requires authentication. Please set your HuggingFace access token as an environment variable."
                )
            elif response.status_code == 404:
                raise AquaValueError(f"Model '{model_id}' not found on HuggingFace.")
            elif response.status_code != 200:
                raise AquaValueError(
                    f"Failed to fetch config for '{model_id}'. Status: {response.status_code}"
                )
            return response.json()
        except requests.RequestException as e:
            raise AquaValueError(
                f"Network error fetching config for {model_id}: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise AquaValueError(
                f"Invalid config format for model '{model_id}'."
            ) from e


class AquaShapeRecommend:
    """
    Interface for recommending GPU shapes for machine learning model deployments
    on Oracle Cloud Infrastructure Data Science service.

    This class provides methods to recommend deployment shapes based on a model's requirements,
    handle recommendation details and troubleshooting, and retrieve specific OCI Machine Learning shapes.
    Must be used within a properly configured and authenticated OCI environment.
    """

    def which_shapes(
        self, request: RequestRecommend
    ) -> Union[ShapeRecommendationReport, Table]:
        """
        Lists valid GPU deployment shapes for the provided model and configuration.

        Validates input, retrieves the model configuration, checks the requested sequence length,
        identifies available and valid compute shapes, and summarizes which shapes are compatible
        with the current model settings.

        Parameters
        ----------
        ocid : str
           OCID of the model to recommend feasible compute shapes.

        available_shapes : List[ComputeShapeSummary]
            List of available shapes to recommend

        generate_table : bool
            whether to generate a rich diff Table or ShapeRecommendationReport (see Returns section)

        Returns
        -------
        Table (generate_table = True)
            A table format for the recommendation report with compatible deployment shapes
            or troubleshooting info citing the largest shapes if no shape is suitable.

        ShapeRecommendationReport (generate_table = False)
            A recommendation report with compatible deployment shapes, or troubleshooting info
            citing the largest shapes if no shape is suitable.

        Raises
        ------
        AquaValueError
            If parameters are missing or invalid, or if no valid sequence length is requested.
        """
        try:
            shapes = self.valid_compute_shapes(compartment_id=request.compartment_id)
            data, model_name = self._get_model_config_and_name(
                request.model_id, request.compartment_id
            )
            llm_config = LLMConfig.from_raw_config(data)
            shape_recommendation_report = self._summarize_shapes_for_seq_lens(
                llm_config, shapes, model_name
            )

            if request.generate_table and shape_recommendation_report.recommendations:
                shape_recommendation_report = self._rich_diff_table(
                    shape_recommendation_report
                )

        # custom error to catch model incompatibility issues
        except AquaRecommendationError as error:
            return ShapeRecommendationReport(
                recommendations=[], troubleshoot=str(error)
            )

        except ValidationError as ex:
            custom_errors = build_pydantic_error_message(ex)
            raise AquaValueError(
                f"Invalid parameters to read config.json of LLM Artifact. Error details: {custom_errors}."
            ) from ex
        except AquaValueError as ex:
            logger.error(f"Error with LLM config: {ex}")
            raise AquaValueError(  # noqa: B904
                f"An error occured while producing recommendations: {ex}"
            )

        return shape_recommendation_report

    def _get_model_config_and_name(
        self, model_id: str, compartment_id: str
    ) -> (dict, str):
        """
        Loads model configuration, handling OCID and Hugging Face model IDs.
        """
        if HuggingFaceModelFetcher.is_huggingface_model_id(model_id):
            logger.info(f"'{model_id}' identified as a Hugging Face model ID.")
            ds_model = self._search_model_in_catalog(model_id, compartment_id)
            if ds_model and ds_model.artifact:
                logger.info(
                    "Loading configuration from existing model catalog artifact."
                )
                try:
                    return (
                        load_config(ds_model.artifact, "config.json"),
                        ds_model.display_name,
                    )
                except AquaFileNotFoundError:
                    logger.warning(
                        "config.json not found in artifact, fetching from Hugging Face Hub."
                    )
            return HuggingFaceModelFetcher.fetch_config_only(model_id), model_id
        else:
            logger.info(f"'{model_id}' identified as a model OCID.")
            ds_model = self._validate_model_ocid(model_id)
            return self._get_model_config(ds_model), ds_model.display_name

    def _search_model_in_catalog(
        self, model_id: str, compartment_id: str
    ) -> Optional[DataScienceModel]:
        """
        Searches for a Hugging Face model in the Data Science model catalog by display name.
        """
        try:
            # This should work since the SDK's list method can filter by display_name.
            models = DataScienceModel.list(
                compartment_id=compartment_id, display_name=model_id
            )
            if models:
                logger.info(f"Found model '{model_id}' in the Data Science catalog.")
                return models[0]
        except Exception as e:
            logger.warning(f"Could not search for model '{model_id}' in catalog: {e}")
        return None

    def valid_compute_shapes(
        self, compartment_id: Optional[str] = None
    ) -> List["ComputeShapeSummary"]:
        """
        Returns a filtered list of GPU-only ComputeShapeSummary objects by reading and parsing a JSON file.

        Parameters
        ----------
        file : str
            Path to the JSON file containing shape data.

        Returns
        -------
        List[ComputeShapeSummary]
            List of ComputeShapeSummary objects passing the checks.

        Raises
        ------
        AquaValueError
            If a compartment_id is not provided and cannot be found in the
            environment variables.
        """
        if not compartment_id:
            compartment_id = os.environ.get(
                "NB_SESSION_COMPARTMENT_OCID"
            ) or os.environ.get("PROJECT_COMPARTMENT_OCID")
            if compartment_id:
                logger.info(f"Using compartment_id from environment: {compartment_id}")

        if not compartment_id:
            raise AquaValueError(
                "A compartment OCID is required to list available shapes. "
                "Please provide it as a parameter or set the 'NB_SESSION_COMPARTMENT_OCID' "
                "or 'PROJECT_COMPARTMENT_OCID' environment variable."
            )

        oci_shapes = OCIDataScienceModelDeployment.shapes(compartment_id=compartment_id)
        set_user_shapes = {shape.name: shape for shape in oci_shapes}

        gpu_shapes_metadata = load_gpu_shapes_index().shapes

        valid_shapes = []
        # only loops through GPU shapes, update later to include CPU shapes
        for name, spec in gpu_shapes_metadata.items():
            if name in set_user_shapes:
                oci_shape = set_user_shapes.get(name)

                compute_shape = ComputeShapeSummary(
                    available=True,
                    core_count=oci_shape.core_count,
                    memory_in_gbs=oci_shape.memory_in_gbs,
                    shape_series=SHAPE_MAP.get(oci_shape.shape_series, "GPU"),
                    name=oci_shape.name,
                    gpu_specs=spec,
                )
            else:
                compute_shape = ComputeShapeSummary(
                    available=False, name=name, shape_series="GPU", gpu_specs=spec
                )
            valid_shapes.append(compute_shape)

        valid_shapes.sort(
            key=lambda shape: shape.gpu_specs.gpu_memory_in_gbs, reverse=True
        )
        return valid_shapes

    @staticmethod
    def _rich_diff_table(shape_report: ShapeRecommendationReport) -> Table:
        """
        Generates a rich-formatted table comparing deployment recommendations
        from a ShapeRecommendationReport object.

        Args:
            shape_report (ShapeRecommendationReport): The report containing shape recommendations.

        Returns:
            Table: A rich Table displaying model deployment recommendations.
        """
        logger.debug(
            "Starting to generate rich diff table from ShapeRecommendationReport."
        )

        name = shape_report.display_name
        header = (
            f"Model Deployment Recommendations: {name}"
            if name
            else "Model Deployment Recommendations"
        )
        logger.debug(f"Table header set to: {header!r}")

        if shape_report.troubleshoot:
            header = f"{header}\n{shape_report.troubleshoot}"
            logger.debug("Appended troubleshoot message to the header.")

        term_columns = shutil.get_terminal_size((120, 20)).columns

        recs_width = min(term_columns - 50, 60)
        logger.debug(f"Calculated recommendation column width: {recs_width}")

        table = Table(
            title=header,
            show_lines=True,
        )
        logger.debug("Initialized Table object.")

        table.add_column("Shape Name", max_width=16)
        table.add_column("Avaliable", max_width=7)
        table.add_column("Shape Type", max_width=7)
        table.add_column("GPU Count", max_width=7)
        table.add_column("Total Memory (GB)", max_width=10)
        table.add_column("Model Deployment Size (GB)", max_width=7)
        table.add_column("Deployment Quantization", max_width=10)
        table.add_column("Recommendation", max_width=recs_width)
        logger.debug("Added table columns with specified max widths.")

        recs = getattr(shape_report, "recommendations", [])
        logger.debug(f"Number of recommendations: {len(recs)}")

        for entry in recs:
            shape = entry.shape_details
            gpu = shape.gpu_specs
            conf = entry.configurations[0]
            model = conf.model_details
            deploy = conf.deployment_params
            recommendation = conf.recommendation

            if deploy.params:
                recommendation = (
                    f"Suggested PARAMS: {deploy.params}\n\n" + recommendation
                )

            if gpu.gpu_memory_in_gbs and shape.memory_in_gbs:
                total_memory = f"GPU: {str(gpu.gpu_memory_in_gbs)}\nCPU: {str(shape.memory_in_gbs)}"
            elif gpu.gpu_memory_in_gbs:
                total_memory = f"GPU: {str(gpu.gpu_memory_in_gbs)}"
            else:
                total_memory = f"CPU: {str(shape.memory_in_gbs)}"

            table.add_row(
                shape.name,
                str(shape.available),
                str(shape.shape_series),
                str(gpu.gpu_count),
                total_memory,
                str(model.total_model_gb),
                deploy.quantization,
                recommendation,
            )

        logger.debug("Completed populating table with recommendation rows.")
        return table

    @staticmethod
    def _validate_model_ocid(ocid: str) -> DataScienceModel:
        """
        Ensures the OCID passed is valid for referencing a DataScienceModel resource.
        """
        resource_type = get_resource_type(ocid)

        if resource_type != "datasciencemodel":
            raise AquaValueError(
                f"The provided OCID '{ocid}' is not a valid Oracle Cloud Data Science Model OCID. "
                "Please provide an OCID corresponding to a Data Science model resource. "
                "Tip: Data Science model OCIDs typically start with 'ocid1.datasciencemodel...'."
            )

        model = DataScienceModel.from_id(ocid)
        return model

    @staticmethod
    def _get_model_config(model: DataScienceModel):
        """
        Loads the configuration for a given Oracle Cloud Data Science model.

        Validates the resource type associated with the provided OCID, ensures the model
        is for text-generation with a supported decoder-only architecture, and loads the model's
        configuration JSON from the artifact path.

        Parameters
        ----------
        model : DataScienceModel
            The DataScienceModel representation of the model used in recommendations

        Returns
        -------
        dict
            The parsed configuration dictionary from config.json.

        Raises
        ------
        AquaValueError
            If the OCID is not for a Data Science model, or if the model type is not supported,
            or if required files/tags are not present.

        AquaRecommendationError
            If the model OCID provided is not supported (only text-generation decoder models in safetensor format supported).
        """

        model_task = model.freeform_tags.get("task", "").lower()
        model_format = model.freeform_tags.get("model_format", "").lower()

        logger.info(f"Current model task type: {model_task}")
        logger.info(f"Current model format: {model_format}")

        if TEXT_GENERATION not in model_task:
            raise AquaRecommendationError(
                "Please provide a decoder-only text-generation model (ex. Llama, Falcon, etc.). "
                f"Only text-generation models are supported in this tool at this time. Current model task type: {model_task}"
            )
        if SAFETENSORS not in model_format:
            msg = "Please provide a model in Safetensor format. "
            if model_format:
                msg += f"The current model format ({model_format}) is not supported by this tool at this time."

            raise AquaRecommendationError(msg)

        if not model.artifact:
            raise AquaValueError(
                "Unable to retrieve model artifact. Ensure model is registered and active."
            )

        try:
            data = load_config(model.artifact, "config.json")

        except AquaFileNotFoundError as e:
            logger.error(
                f"config.json not found in model artifact at {model.artifact}: {e}"
            )
            raise AquaRecommendationError(
                "The configuration file 'config.json' was not found in the specified model directory. "
                "Please ensure your model follows the Hugging Face format and includes a 'config.json' with the necessary architecture parameters."
            ) from e

        return data

    @staticmethod
    def _summarize_shapes_for_seq_lens(
        config: LLMConfig,
        shapes: List[ComputeShapeSummary],
        name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """
        Generate a recommendation report for eligible deployment shapes by evaluating
        model memory consumption and maximum model length for given configurations.

        Parameters
        ----------
        config : LLMConfig
            The loaded model configuration.
        shapes : List[ComputeShapeSummary]
            All candidate deployment shapes.
        name : str
            name of the model
        batch_size : int, optional
            Batch size to evaluate (default is 1).

        Returns
        -------
        ShapeRecommendationReport
            Report containing shape recommendations and troubleshooting advice, if any.

        Raises
        ------
        ValueError
            If no GPU shapes are available.

        Notes
        -----
        - Considers quantization if defined in config, otherwise cycles through optimal configs.
        - Applies pareto optimality if too many recommendations.
        - Provides troubleshooting options if nothing fits.
        """
        recommendations = []

        if not shapes:
            raise AquaValueError(
                "No GPU shapes were passed for recommendation. Ensure shape parsing succeeded."
            )

        # Pre-quantized: only consider different max-seq-len
        if config.quantization_type:
            deployment_config = config.calculate_possible_seq_len()
            for shape in shapes:
                shape_quantization = set(shape.gpu_specs.quantization)
                if config.quantization_type in shape_quantization:
                    allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                    for max_seq_len in deployment_config:
                        estimator = get_estimator(
                            llm_config=config,
                            seq_len=max_seq_len,
                            batch_size=batch_size,
                        )
                        if estimator.validate_shape(allowed_gpu_memory):
                            best_config = [
                                ModelConfig.constuct_model_config(
                                    estimator, allowed_gpu_memory
                                )
                            ]
                            recommendations.append(
                                ShapeReport(
                                    shape_details=shape, configurations=best_config
                                )
                            )
                            break

        # unquantized: consider inflight quantization (4bit)
        else:
            deployment_config = config.optimal_config()
            prev_quant = None
            for shape in shapes:
                shape_quantization = set(shape.gpu_specs.quantization)
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                for quantization, max_seq_len in deployment_config:
                    if (
                        quantization == BITS_AND_BYTES_4BIT
                        and BITSANDBYTES not in shape_quantization
                    ):
                        continue
                    if quantization != prev_quant:
                        updated_config = config.model_copy(
                            update={"in_flight_quantization": quantization}
                        )
                        prev_quant = quantization
                    estimator = get_estimator(
                        llm_config=updated_config,
                        seq_len=max_seq_len,
                        batch_size=batch_size,
                    )
                    if estimator.validate_shape(allowed_gpu_memory):
                        best_config = [
                            ModelConfig.constuct_model_config(
                                estimator, allowed_gpu_memory
                            )
                        ]
                        recommendations.append(
                            ShapeReport(shape_details=shape, configurations=best_config)
                        )
                        break

        troubleshoot_msg = ""

        if len(recommendations) > 2:
            recommendations = ShapeReport.pareto_front(recommendations)

        if not recommendations:
            # Troubleshooting advice if nothing fits
            # Assumes shapes is sorted largest to smallest and quantizations 'fp8'/'4bit' exist
            troubleshoot_msg += TROUBLESHOOT_MSG

            largest_shapes = (
                [(shapes[0], "fp8", False), (shapes[1], "4bit", True)]
                if len(shapes) > 1
                else []
            )  # shape, quantization, in_flight_quantization

            for shape, quantization, in_flight in largest_shapes:
                if in_flight:
                    updated_config = config.model_copy(
                        update={"in_flight_quantization": quantization}
                    )
                else:
                    updated_config = config.model_copy(
                        update={"quantization": quantization}
                    )
                estimator = get_estimator(
                    llm_config=updated_config, seq_len=2048, batch_size=batch_size
                )
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs * 0.9
                best_config = [
                    ModelConfig.constuct_model_config(estimator, allowed_gpu_memory)
                ]
                recommendations.append(
                    ShapeReport(shape_details=shape, configurations=best_config)
                )

        return ShapeRecommendationReport(
            display_name=name,
            recommendations=recommendations,
            troubleshoot=troubleshoot_msg,
        )
