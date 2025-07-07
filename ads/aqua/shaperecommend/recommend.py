import json
from typing import List

from pydantic import ValidationError

from ads.aqua.app import AquaApp, logger
from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import (
    build_pydantic_error_message,
    get_model_by_reference_paths,
    load_config,
)
from ads.aqua.model.constants import ModelTask
from ads.aqua.shaperecommend.constants import SHAPES_METADATA
from ads.aqua.shaperecommend.estimator import get_estimator
from ads.aqua.shaperecommend.llm_config import LLMConfig
from ads.aqua.shaperecommend.shape_report import (
    ModelConfig,
    RequestRecommend,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.model.datascience_model import DataScienceModel


class AquaRecommendApp(AquaApp):
    """
    Interface for recommending GPU shapes for machine learning model deployments
    on Oracle Cloud Infrastructure Data Science service.

    This class provides methods to recommend deployment shapes based on a model's requirements,
    handle recommendation details and troubleshooting, and retrieve specific OCI Machine Learning shapes.
    Must be used within a properly configured and authenticated OCI environment.

    Methods
    -------
    which_gpu(self, **kwargs) -> List[Dict]:
        Lists the valid GPU deployment shapes that fit the given model and user-provided settings.

    Note:
        Use `ads aqua recommend which_gpu --help` to get more details on available parameters.
    """

    def which_gpu(self, **kwargs) -> ShapeRecommendationReport:
        """
        Lists valid GPU deployment shapes for the provided model and configuration.

        Validates input, retrieves the model configuration, checks the requested sequence length,
        identifies available and valid compute shapes, and summarizes which shapes are compatible
        with the current model settings.

        Parameters
        ----------
        model : str
            Name of the model to deploy.
        max_model_len : int, optional
            Maximum sequence length/user context length the model should support.

        Returns
        -------
        ShapeRecommendationReport
            A recommendation report with compatible deployment shapes, or troubleshooting info
            if no shape is suitable.

        Raises
        ------
        AquaValueError
            If parameters are missing or invalid, or if no valid sequence length is requested.
        """
        try:
            request = RequestRecommend(**kwargs)
            model = DataScienceModel.from_id(request.model_ocid)

            if ModelTask.TEXT_GENERATION not in model.freeform_tags:
                AquaValueError()

            model.artifact

            config = load_config()

        except ValidationError as ex:
            custom_errors = build_pydantic_error_message(ex)
            raise AquaValueError(
                f"Invalid parameters for creating a model deployment. Error details: {custom_errors}."
            ) from ex

        available_shapes = self.valid_compute_shapes()

        return self.summarize_shapes_for_seq_lens(
            config, available_shapes
        )

    @staticmethod
    def valid_compute_shapes(file: str = SHAPES_METADATA) -> List["ComputeShapeSummary"]:
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
        ValueError
            If the file cannot be opened, parsed, or the 'shapes' key is missing.
        """
        try:
            with open(file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read or parse shapes JSON file '{file}': {e}")  # noqa: B904

        if 'shapes' not in data or not isinstance(data['shapes'], dict):
            raise ValueError(f"'shapes' key is missing or invalid in the JSON file: {file}")

        shapes = data['shapes']
        valid_shapes = []
        for name, spec in shapes.items():
            valid_shapes.append(ComputeShapeSummary(name=name, shape_series="GPU", gpu_specs=spec))
        valid_shapes.sort(key=lambda shape: shape.gpu_specs.gpu_memory_in_gbs, reverse=True)
        return valid_shapes

    @staticmethod
    def summarize_shapes_for_seq_lens(
        config: LLMConfig,
        shapes: List[ComputeShapeSummary],
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
            raise ValueError("No GPU shapes were passed for recommendation. Ensure shape parsing succeeded.")

        # Pre-quantized: only consider different max-seq-len
        if config.quantization_type:
            deployment_config = config.calculate_possible_seq_len()
            for shape in shapes:
                if config.quantization_type in shape.gpu_specs.quantization:
                    allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                    for max_seq_len in deployment_config:
                        estimator = get_estimator(config=config, seq_len=max_seq_len, batch_size=batch_size)
                        if estimator.validate_shape(estimator, allowed_gpu_memory):
                            best_config = [ModelConfig.constuct_model_config(estimator, allowed_gpu_memory)]
                            recommendations.append(ShapeReport(shape_details=shape, configurations=best_config))
                            break

        # unquantized: consider inflight quantization (4bit and 8bit)
        else:
            deployment_config = config.optimal_config()
            prev_quant = None
            for shape in shapes:
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                for quantization, max_seq_len in deployment_config:
                    if quantization != prev_quant:
                        updated_config = config.model_copy(update={"quantization": quantization})
                        prev_quant = quantization
                    estimator = get_estimator(config=updated_config, seq_len=max_seq_len, batch_size=batch_size)
                    if estimator.validate_shape(allowed_gpu_memory):
                        best_config = [ModelConfig.constuct_model_config(estimator, allowed_gpu_memory)]
                        recommendations.append(ShapeReport(shape_details=shape, configurations=best_config))
                        break

        troubleshoot = []
        if len(recommendations) > 5:
            recommendations = ShapeReport.pareto_front(recommendations)

        if not recommendations:
            # Troubleshooting advice if nothing fits
            # Assumes shapes is sorted largest to smallest and quantizations 'fp8'/'4bit' exist
            largest_shapes = [(shapes[0], "fp8"), (shapes[1], "4bit")] if len(shapes) > 1 else []
            for shape, quantization in largest_shapes:
                updated_config = config.model_copy(update={"quantization": quantization})
                estimator = get_estimator(config=updated_config, seq_len=2048, batch_size=batch_size)
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs * 0.9
                best_config = [ModelConfig.constuct_model_config(estimator, allowed_gpu_memory)]
                troubleshoot.append(ShapeReport(shape_details=shape, configurations=best_config))

        return ShapeRecommendationReport(
            recommendations=recommendations,
            troubleshoot=troubleshoot
        )