#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import re
import shutil
from typing import Dict, List, Optional, Tuple, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError
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
    format_hf_custom_error_message,
    get_resource_type,
    is_valid_ocid,
    load_config,
    load_gpu_shapes_index,
)
from ads.aqua.shaperecommend.constants import (
    BITS_AND_BYTES_4BIT,
    BITSANDBYTES,
    SAFETENSORS,
    SHAPE_MAP,
    TEXT_GENERATION,
    TROUBLESHOOT_MSG,
)
from ads.aqua.shaperecommend.estimator import get_estimator
from ads.aqua.shaperecommend.llm_config import LLMConfig
from ads.aqua.shaperecommend.shape_report import (
    ModelConfig,
    RequestRecommend,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.config import COMPARTMENT_OCID
from ads.model.datascience_model import DataScienceModel
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)


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

        This method validates input, retrieves the model configuration, checks the requested sequence length,
        identifies available and valid compute shapes, and summarizes which shapes are compatible
        with the current model settings.

        Parameters
        ----------
        request : RequestRecommend
            The request object with all needed recommendation fields:
            model_id : str
                OCID of the model to recommend feasible compute shapes for.
            generate_table : bool, optional
                If True (default), generate a rich diff table as output. If False, return a ShapeRecommendationReport object.
            deployment_config : Optional[AquaDeploymentConfig]
                Deployment configuration for the model (used for service models only).
            compartment_id : str, optional
                The OCID of the user's compartment (needed if shape availability is compartment-specific).

        Returns
        -------
        Table
            If `generate_table` is True, returns a table with the recommendation report, listing compatible deployment shapes or troubleshooting info citing the largest shapes if no shape is suitable.
        ShapeRecommendationReport
            If `generate_table` is False, returns a recommendation report with compatible deployment shapes, or troubleshooting info if no shape is suitable.

        Raises
        ------
        AquaValueError
            If required parameters are missing or invalid, or if no valid sequence length is available.
        """
        try:
            shapes = self.valid_compute_shapes(compartment_id=request.compartment_id)

            if request.deployment_config:
                if is_valid_ocid(request.model_id):
                    ds_model = self._get_data_science_model(request.model_id)
                    model_name = ds_model.display_name
                else:
                    model_name = request.model_id

                shape_recommendation_report = (
                    ShapeRecommendationReport.from_deployment_config(
                        request.deployment_config, model_name, shapes
                    )
                )

            else:
                data, model_name = self._get_model_config_and_name(
                    model_id=request.model_id,
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
        self,
        model_id: str,
    ) -> Tuple[Dict, str]:
        """
        Loads model configuration by trying OCID logic first, then falling back
        to treating the model_id as a Hugging Face Hub ID.

        Parameters
        ----------
        model_id : str
            The model OCID or Hugging Face model ID.
        # compartment_id : Optional[str]
        #     The compartment OCID, used for searching the model catalog.

        Returns
        -------
        Tuple[Dict, str]
            A tuple containing:
            - The model configuration dictionary.
            - The display name for the model.
        """
        if is_valid_ocid(model_id):
            logger.info(f"Detected OCID: Fetching OCI model config for '{model_id}'.")
            ds_model = self._get_data_science_model(model_id)
            config = self._get_model_config(ds_model)
            model_name = ds_model.display_name
        else:
            logger.info(
                f"Assuming Hugging Face model ID: Fetching config for '{model_id}'."
            )
            config = self._fetch_hf_config(model_id)
            model_name = model_id

        return config, model_name

    def _fetch_hf_config(self, model_id: str) -> Dict:
            """
            Downloads a model's config.json from Hugging Face Hub.
            """
            try:
                config_path = hf_hub_download(repo_id=model_id, filename="config.json")
                with open(config_path, encoding="utf-8") as f:
                    return json.load(f)

            except EntryNotFoundError as e:
                # EXPLICIT HANDLING: This covers the GGUF case
                logger.error(f"config.json not found for model '{model_id}': {e}")
                raise AquaRecommendationError(
                    f"The configuration file 'config.json' was not found in the repository '{model_id}'. "
                    "This often happens with GGUF models (which are not supported) or invalid repositories. "
                    "Please ensure the model ID is correct and the repository contains a 'config.json'."
                ) from e

            except HfHubHTTPError as e:
                # For other errors (Auth, Network), use the shared formatter.
                logger.error(f"HTTP error fetching config for '{model_id}': {e}")
                format_hf_custom_error_message(e) 
                
            except Exception as e:
                logger.error(f"Unexpected error fetching config for '{model_id}': {e}")
                raise AquaRecommendationError(
                    f"An unexpected error occurred while fetching the model configuration: {e}"
                ) from e
                
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
            compartment_id = COMPARTMENT_OCID
            if compartment_id:
                logger.info(f"Using compartment_id from environment: {compartment_id}")

        if not compartment_id:
            raise AquaValueError(
                "A compartment OCID is required to list available shapes. "
                "Please specify it using the --compartment_id parameter.\n\n"
                "Example:\n"
                'ads aqua deployment recommend_shape --model_id "<YOUR_MODEL_OCID>" --compartment_id "<YOUR_COMPARTMENT_OCID>"'
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

        header = (
            f"{header}\n"
            "Currently, only the VLLM container is supported. "
            "All shape and parameter recommendations will be generated for the VLLM container."
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

            model_size = str(model.total_model_gb) if model else "-"
            quantization = (
                deploy.quantization or deploy.weight_dtype
                if deploy.quantization or deploy.weight_dtype
                else "-"
            )

            table.add_row(
                shape.name,
                str(shape.available),
                str(shape.shape_series),
                str(gpu.gpu_count),
                total_memory,
                model_size,
                quantization,
                recommendation,
            )

        logger.debug("Completed populating table with recommendation rows.")
        return table

    @staticmethod
    def _get_data_science_model(ocid: str) -> DataScienceModel:
        """
        Ensures the OCID passed is valid for referencing a DataScienceModel resource.
        If valid OCID, returns the DataScienceModel
        """
        resource_type = get_resource_type(ocid)

        if resource_type != "datasciencemodel":
            raise AquaValueError(
                f"The provided OCID '{ocid}' is not a valid Oracle Cloud Data Science Model OCID. "
                "Please provide an OCID corresponding to a Data Science model resource. "
                "Tip: Data Science model OCIDs typically start with 'ocid1.datasciencemodel...'."
            )

        return DataScienceModel.from_id(ocid)

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
        model_task = re.sub(r"-", "_", model_task)
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
