import shutil
from typing import List

from pydantic import ValidationError
from rich.table import Table

from ads.aqua.app import AquaApp, logger
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
)
from ads.aqua.shaperecommend.constants import (
    SAFETENSORS,
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
        model_ocid : str
           OCID of the model to recommend feasible compute shapes.

        Returns
        -------
        ShapeRecommendationReport
            A recommendation report with compatible deployment shapes, or troubleshooting info
            citing the largest shapes if no shape is suitable.

        Raises
        ------
        AquaValueError
            If parameters are missing or invalid, or if no valid sequence length is requested.
        """
        try:
            request = RequestRecommend(**kwargs)
            data, model_name = self.get_model_config(request.model_ocid)

            llm_config = LLMConfig.from_raw_config(data)

            available_shapes = self.valid_compute_shapes()
            recommendations = self.summarize_shapes_for_seq_lens(
                llm_config, available_shapes, model_name
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
            raise

        return recommendations

    @staticmethod
    def rich_diff_table(shape_report: ShapeRecommendationReport) -> Table:
        """
        Generates a rich-formatted table comparing deployment recommendations
        from a ShapeRecommendationReport object.

        Args:
            shape_report (ShapeRecommendationReport): The report containing shape recommendations.

        Returns:
            Table: A rich Table displaying model deployment recommendations.
        """
        logger.debug("Starting to generate rich diff table from ShapeRecommendationReport.")

        name = shape_report.model_name
        header = f"Model Deployment Recommendations: {name}" if name else "Model Deployment Recommendations"
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
        table.add_column("GPU Count", max_width=7)
        table.add_column("Total GPU Memory (GB)", max_width=7)
        table.add_column("Model Size (GB)", max_width=7)
        table.add_column("KV Cache Size (GB)", max_width=7)
        table.add_column("Total Model (GB)", max_width=7)
        table.add_column("Deployment Quantization", max_width=10)
        table.add_column("Max Model Length", max_width=7)
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
            full_recommendation = conf.recommendation

            table.add_row(
                shape.name,
                str(gpu.gpu_count),
                str(gpu.gpu_memory_in_gbs),
                str(model.model_size_gb),
                str(model.kv_cache_size_gb),
                str(model.total_model_gb),
                deploy.quantization,
                str(deploy.max_model_len),
                full_recommendation
            )

        logger.debug("Completed populating table with recommendation rows.")
        return table


    def shapes(self, **kwargs) -> Table:
        """
        For the CLI, generates the table (in rich diff) with valid GPU deployment shapes
        for the provided model and configuration.

        Validates if recommendations are generated, calls method to construct the rich diff
        table with the recommendation data.

        Parameters
        ----------
        model_ocid : str
           OCID of the model to recommend feasible compute shapes.

        Returns
        -------
        Table
            A table format for the recommendation report with compatible deployment shapes
            or troubleshooting info citing the largest shapes if no shape is suitable.

        Raises
        ------
        AquaValueError
            If model type is unsupported by tool (no recommendation report generated)
        """
        shape_recommend_report = self.which_gpu(**kwargs)
        if not shape_recommend_report.recommendations:
            if shape_recommend_report.troubleshoot:
                raise AquaValueError(shape_recommend_report.troubleshoot)
            else:
                raise AquaValueError("Unable to generate recommendations from model. Please ensure model is registered and is a decoder-only text-generation model.")

        return self.rich_diff_table(shape_recommend_report)

    @staticmethod
    def get_model_config(ocid: str):
        """
        Loads the configuration for a given Oracle Cloud Data Science model.

        Validates the resource type associated with the provided OCID, ensures the model
        is for text-generation with a supported decoder-only architecture, and loads the model's
        configuration JSON from the artifact path.

        Parameters
        ----------
        ocid : str
            The OCID of the Data Science model.

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
        resource_type = get_resource_type(ocid)

        if resource_type != "datasciencemodel":
            raise AquaValueError(
                f"The provided OCID '{ocid}' is not a valid Oracle Cloud Data Science Model OCID. "
                "Please provide an OCID corresponding to a Data Science model resource. "
                "Tip: Data Science model OCIDs typically start with 'ocid1.datasciencemodel...'."
            )

        model = DataScienceModel.from_id(ocid)

        model_name = model.display_name

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
            msg = "Please provide a model in Safetensor format."
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

        return data, model_name

    @staticmethod
    def valid_compute_shapes() -> List["ComputeShapeSummary"]:
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
        gpu_shapes_metadata = load_gpu_shapes_index().shapes

        valid_shapes = []
        for name, spec in gpu_shapes_metadata.items():
            valid_shapes.append(
                ComputeShapeSummary(name=name, shape_series="GPU", gpu_specs=spec)
            )
        valid_shapes.sort(
            key=lambda shape: shape.gpu_specs.gpu_memory_in_gbs, reverse=True
        )
        return valid_shapes

    @staticmethod
    def summarize_shapes_for_seq_lens(
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
            raise ValueError(
                "No GPU shapes were passed for recommendation. Ensure shape parsing succeeded."
            )

        # Pre-quantized: only consider different max-seq-len
        if config.quantization_type:
            deployment_config = config.calculate_possible_seq_len()
            for shape in shapes:
                if config.quantization_type in shape.gpu_specs.quantization:
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

        # unquantized: consider inflight quantization (4bit and 8bit)
        else:
            deployment_config = config.optimal_config()
            prev_quant = None
            for shape in shapes:
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                for quantization, max_seq_len in deployment_config:
                    if quantization != prev_quant:
                        updated_config = config.model_copy(
                            update={"quantization": quantization}
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
                [(shapes[0], "fp8"), (shapes[1], "4bit")] if len(shapes) > 1 else []
            )
            for shape, quantization in largest_shapes:
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
            model_name=name, recommendations=recommendations, troubleshoot=troubleshoot_msg
        )
