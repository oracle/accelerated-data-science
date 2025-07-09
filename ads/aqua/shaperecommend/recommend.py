import json
from typing import List

from huggingface_hub import hf_hub_download
from pydantic import ValidationError

from ads.aqua.app import AquaApp, logger
from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import build_pydantic_error_message, list_hf_models
from ads.aqua.modeldeployment.deployment import AquaDeploymentApp
from ads.aqua.shaperecommend.constants import NEXT_QUANT
from ads.aqua.shaperecommend.estimator import MemoryEstimator, get_estimator
from ads.aqua.shaperecommend.llm_config import LLMConfig
from ads.aqua.shaperecommend.shape_report import (
    DeploymentShapeSummary,
    GPUSummary,
    RequestRecommend,
    ShapeRecommendationReport,
    ShapeSummary,
    TroubleshootShapeSummary,
)


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
            config = self.get_model_config(request.model)
        except ValidationError as ex:
            custom_errors = build_pydantic_error_message(ex)
            raise AquaValueError(
                f"Invalid parameters for creating a model deployment. Error details: {custom_errors}."
            ) from ex

        valid_seq_lens = self.power_of_two_seq_lens(max_len=config.max_seq_len)
        if request.max_model_len not in valid_seq_lens:
            valid_seq_lens_str = " ".join(map(str, valid_seq_lens))
            raise AquaValueError(
                f"Invalid model sequence length requested. Please select one model sequence length: {valid_seq_lens_str}"
            )

        available_shapes = AquaDeploymentApp().list_shapes()
        valid_gpu_shapes = self.valid_compute_shapes(available_shapes)
        return self.summarize_shapes_for_seq_lens(
            config, valid_gpu_shapes, user_seq_len=request.max_model_len
        )

    def get_model_config(self, model_name: str) -> LLMConfig:
        """
        Downloads config.json for a model from Hugging Face and parses it into an LLMConfig instance.
        Handles errors gracefully if the model or config cannot be retrieved.
        """

        model_ids = list_hf_models(model_name)
        if not model_ids:
            raise AquaValueError(
                f"No models found for your query: '{model_name}'."
            )

        model_id = model_ids[0]  # Select the first model from the list

        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
            return LLMConfig(**config_data)

        except Exception as ex:
            raise AquaValueError(
                f"Error retrieving or parsing config.json for model '{model_name}': {ex}"
            ) from ex

    @staticmethod
    def valid_compute_shapes(
        compute_shapes: List["ComputeShapeSummary"]
    ) -> List["ComputeShapeSummary"]:
        """
        Returns a filtered list of ComputeShapeSummary objects that are considered valid.

        A shape is valid if:
        - It has a non-empty name,
        - gpu_specs is present,
        - gpu_memory_in_gbs and gpu_count are present in gpu_specs.

        Args:
            compute_shapes: List of ComputeShapeSummary objects to validate.

        Returns:
            List of ComputeShapeSummary objects passing the above checks.
        """
        return [
            shape
            for shape in compute_shapes
            if shape.name
            and getattr(shape, "gpu_specs", None)
            and getattr(shape.gpu_specs, "gpu_memory_in_gbs", None)
            and getattr(shape.gpu_specs, "gpu_count", None)
        ]

    @staticmethod
    def power_of_two_seq_lens(min_len=2048, max_len=16384) -> List[int]:
        """
        Calculates the range of valid sequence lengths (power of two) up until
        the model's max sequence length as specified in the LLMConfig.
        """
        vals = []
        curr = min_len
        while curr <= max_len:
            vals.append(curr)
            curr *= 2
        if vals[-1] != max_len:
            vals.append(max_len)
        return vals

    def suggest_param_advice(self, estimator: MemoryEstimator, allowed) -> str:
        """
        Returns a tailored suggestion on how the user should improve the memory footprint of their model.
        Identifies whether the KV cache and/or model size is the bottleneck for memory footprint.
        """
        kv_gb = estimator.kv_cache_memory
        wt_gb = estimator.model_memory
        batch_size = estimator.batch_size
        seq_len = estimator.seq_len
        weight_size = estimator.config.weight_dtype
        suggested_quant_msg = None

        if estimator.config.suggested_quantizations:
            quant_advice = ", ".join(estimator.config.suggested_quantizations)
            suggested_quant_msg = f"Use the same model with {quant_advice} quantization"

        kv_advice = (
            f"To reduce KV cache memory usage: \n"
            f"1. reduce maximum context length (set --max-model-len to less than current max sequence length: {seq_len})\n"
            f"2. reduce batch size to less than {batch_size}."
            if batch_size > 1
            else ".\n"
        )

        wt_advice = (
            f"To reduce model size:\n"
            f"1. Consider using a model with fewer parameters. \n"
            f"2. {suggested_quant_msg or 'a quantized version (e.g., INT8 or another supported type)'}, which is smaller than the current quantization/ weight size: {estimator.config.quantization if estimator.config.quantization in NEXT_QUANT.keys() else weight_size}."
        )

        if kv_gb > wt_gb and kv_gb > allowed * 0.5:
            # KV cache drives memory usage
            main = "KV cache memory usage is the bottleneck of memory use."
            advice = kv_advice

        elif wt_gb > kv_gb and wt_gb > allowed * 0.5:
            # model weights drives memory usage
            main = "The model configuration is the bottleneck of memory use."
            advice = wt_advice

        else:
            main = "Both model weights and KV cache are significant contributors to memory use."
            advice = kv_advice + "\n" + wt_advice
        return f"{main} ({kv_gb:.1f}GB KV cache, {wt_gb:.1f}GB weights).\n{advice}"

    def limiting_factor(
        self,
        estimator: MemoryEstimator,
        available_ram,
        gpu_utilization: float,
        warn_delta=0.9,
    ) -> str:
        """
        Warns the user if a certain valid compute shape would be close to the memory limit if model w/ current parameters was used.
        Uses the suggestions from suggest_param_advice to give tailored warnings.
        """
        required = estimator.total_memory
        allowed = available_ram * gpu_utilization

        quantization = getattr(estimator.config, "quantization", "None")
        weight_size = estimator.config.weight_dtype
        batch_size = estimator.batch_size
        seq_len = estimator.seq_len

        param_advice = self.suggest_param_advice(estimator, allowed)

        # even if model configuration works, if we are close to the limit, we should warn user
        if required > allowed * warn_delta:
            advice = (
                f"The selected model configuration is close to GPU Memory Limit ({required:.1f}GB used / {allowed:.1f}GB allowed).\n"
                + param_advice
            )
            return advice
        else:
            return (
                f"Model fits well within limits of compute shape. ({required:.1f}GB used / {allowed:.1f}GB allowed)\n"
                f"(Current batch size: {batch_size}, context length: {seq_len}, "
                f"quantization/ model weight size: {quantization or weight_size})."
            )

    def calc_gpu_report_per_shape(
        self,
        estimator: MemoryEstimator,
        shape: ComputeShapeSummary,
        gpu_utilization: float,
    ) -> ShapeSummary:
        """
        Generate a summary of GPU memory and compute usage for a specific shape configuration.

        For a given compute shape, evaluates all powers-of-two allocations of available GPUs,
        and for each valid configuration (where total available GPU memory exceeds model requirements),
        generates a `GPUSummary` describing per-GPU memory allocation and the system's limiting factor.

        Parameters:
            estimator (MemoryEstimator): The memory estimator object containing model memory requirements.
            shape (ComputeShapeSummary): The compute shape configuration, including GPU specs.
            gpu_utilization (float): The fraction (0.0–1.0) of total GPU memory to consider usable.

        Returns:
            ShapeSummary: A summary object containing the shape name and a list of valid `GPUSummary`
                        entries. Returns `None` if no valid GPU configurations are possible for the shape
                        and utilization provided.
        """
        power = 1

        limit = shape.gpu_specs.gpu_count
        num_gpu_cards = []

        # get eligible number of cards
        while limit and power <= limit:
            num_gpu_cards.append(power)
            power *= 2

        # take gpu_memory_in_gbs / s.gpu_specs.gpu_count -> gpu_memory/ gpu card * used_gps -> available ram
        memory_per_gpu = shape.gpu_specs.gpu_memory_in_gbs / shape.gpu_specs.gpu_count

        gpu_reports = []

        for used_gpus in num_gpu_cards:
            available_ram = used_gpus * memory_per_gpu
            eligible = available_ram * gpu_utilization > estimator.total_memory
            if eligible:
                limit = self.limiting_factor(estimator, available_ram, gpu_utilization)
                gpu_reports.append(
                    GPUSummary(
                        gpu_count=used_gpus,
                        gpu_memory_in_gb=available_ram,
                        limiting_factor=limit,
                    )
                )

        return (
            ShapeSummary(shape=shape.name, gpu_reports=gpu_reports)
            if gpu_reports
            else None
        )

    def summarize_shapes_for_seq_lens(
        self,
        config: LLMConfig,
        shapes: List[ComputeShapeSummary],
        batch_size: int = 1,
        user_seq_len: int = 4096,
        gpu_utilization: float = 0.95,
    ) -> ShapeRecommendationReport:
        """
        Generate a recommendation report for eligible deployment shapes by considering model memory consumption
        and max model length.

        Parameters
        ----------
        config : LLMConfig
            The loaded model config.
        shapes : List[ComputeShapeSummary]
            All candidate deployment shapes.
        batch_size : int
            Batch size to evaluate.
        user_seq_lens : Optional[List[int]]
            Sequence lengths (contexts) provided by the user; if None, use defaults.
        gpu_utilization : float
            Utilization margin (e.g., 0.8, 0.9).

        Returns
        -------
        ShapeRecommendationReport
        """

        recs = []

        shape_reports = []

        estimator = get_estimator(
            config=config, batch_size=batch_size, seq_len=user_seq_len
        )

        logger.info(f"The {type(estimator)} will be used.")

        max_gpu_memory_size = 0

        for shape in shapes:
            shape_report = self.calc_gpu_report_per_shape(
                estimator, shape, gpu_utilization
            )

            if shape_report:
                shape_reports.append(shape_report)

            if (
                shape.gpu_specs
                and shape.gpu_specs.gpu_memory_in_gbs > max_gpu_memory_size
            ):
                # reassign memory shape if we encounter a new larger shape
                max_gpu_memory_shape = shape
                max_gpu_memory_size = shape.gpu_specs.gpu_memory_in_gbs

        recs.append(
            DeploymentShapeSummary(
                batch_size=batch_size,
                precision=config.quantization or config.weight_dtype,
                gb_used_by_model=round(estimator.total_memory, 3),
                max_seq_len=user_seq_len,
                shape_reports=shape_reports,
            )
        )

        # we don't have any compatible shape recommendations but have shapes available in env
        if shapes and not shape_reports:
            # suggest the largest shape w/ actionable advice on how to make it fit
            allowed = max_gpu_memory_size * gpu_utilization
            advice = self.suggest_param_advice(estimator, allowed)

            troubleshoot = TroubleshootShapeSummary(
                largest_shape=max_gpu_memory_shape.name,
                gpu_memory_in_gb=max_gpu_memory_size,
                gb_used_by_model=round(estimator.total_memory, 3),
                max_seq_len=user_seq_len,
                batch_size=batch_size,
                precision=config.quantization or config.weight_dtype,
                advice=advice,
            )

        return ShapeRecommendationReport(
            recommendations=recs, troubleshoot=troubleshoot
        )
