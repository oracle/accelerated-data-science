#!/usr/bin/env python
#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import json
from typing import List, Optional

from pydantic import BaseModel, Field

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.modeldeployment.config_loader import AquaDeploymentConfig
from ads.aqua.shaperecommend.constants import (
    MAX_MODEL_LEN_FLAG,
    QUANT_FLAG,
    QUANT_MAPPING,
    VLLM_ENV,
    VLLM_PARAMS_FAMILY,
    WEIGHT_DTYPE_FLAG,
)
from ads.aqua.shaperecommend.estimator import MemoryEstimator
from ads.config import COMPARTMENT_OCID


class RequestRecommend(BaseModel):
    """
    A request to recommend compute shapes and parameters for a given model.
    """

    model_id: str = Field(
        ...,
        description=(
            "The OCID or Hugging Face ID of the model for which to recommend feasible compute shapes."
        ),
    )

    generate_table: Optional[bool] = Field(
        True,
        description=(
            "If True, generate a rich formatted table as the response. "
            "If False, return the recommendation as a JSON structure."
        ),
    )

    compartment_id: Optional[str] = Field(
        COMPARTMENT_OCID,
        description="The OCID of the user's compartment.",
    )

    deployment_config: Optional["AquaDeploymentConfig"] = Field(
        None,
        description=(
            "The deployment configuration for the model (only available for service models)."
        ),
    )

    class Config:
        protected_namespaces = ()


class DeploymentParams(BaseModel):  # noqa: N801
    """
    Recommended parameters for deployment and model inferencing (specific to compute shape & model).
    """

    params: str = Field(
        ..., description="Runtime parameters for deployment with vLLM, etc."
    )
    quantization: Optional[str] = Field(
        None, description="Type of quantization (e.g. 4bit)."
    )
    weight_dtype: Optional[str] = Field(
        None, description="Data type that the model weights use (bfloat16)."
    )
    max_model_len: Optional[int] = Field(
        None, description="Maximum length of input sequence."
    )
    env_var: Optional[dict] = Field(
        None, description="Global environment variables needed for deployment."
    )


class ModelDetail(BaseModel):
    """
    The estimated memory footprint of a model, KV cache, and its total (model + KV cache).
    """

    model_size_gb: float = Field(..., description="Size of the model in GB.")
    kv_cache_size_gb: float = Field(..., description="Size of KV cache in GB.")
    total_model_gb: float = Field(
        ..., description="Total size of model and cache in GB."
    )

    class Config:
        protected_namespaces = ()


class ModelConfig(BaseModel):
    """
    The configuration for a model based on specific set of deployment parameters and memory capacity of shape.
    """

    deployment_params: DeploymentParams = Field(
        ..., description="Parameters for deployment."
    )
    model_details: Optional[ModelDetail] = Field(
        None, description="Details about the model."
    )

    recommendation: Optional[str] = Field(
        "", description="GPU recommendation for the model."
    )

    class Config:
        protected_namespaces = ()

    @classmethod
    def constuct_model_config(
        cls, estimator: MemoryEstimator, allowed_gpu_memory: float
    ) -> "ModelConfig":
        """
        Assembles a complete ModelConfig, including model details, deployment parameters (vLLM), and recommendations.

        Parameters
        ----------
        shape_quantization : set[str]
            Allowed quantization methods for the compute shape

        Returns
        -------
        ModelConfig
            Contains round-tripped model size, kv cache, total, vLLM parameters, and recommendations.

        Notes
        -----
        - Rounds all sizes to 3 decimal digits.
        - Computes a recommendation string using `limiting_factor`.
        """
        c = estimator.llm_config
        deployment_params = DeploymentParams(
            quantization=c.quantization or c.in_flight_quantization or c.weight_dtype,
            max_model_len=getattr(estimator, "seq_len", None),
            params=estimator.construct_deployment_params(),
        )
        model_detail = ModelDetail(
            model_size_gb=round(getattr(estimator, "model_memory", 0.0), 2),
            kv_cache_size_gb=round(getattr(estimator, "kv_cache_memory", 0.0), 2),
            total_model_gb=round(getattr(estimator, "total_memory", 0.0), 2),
        )
        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=estimator.limiting_factor(allowed_gpu_memory),
        )


class ShapeReport(BaseModel):
    """
    The feasible deployment configurations for the model per shape.
    """

    shape_details: "ComputeShapeSummary" = Field(
        ..., description="Details about the compute shape (ex. VM.GPU.A10.2)."
    )
    configurations: List["ModelConfig"] = Field(
        default_factory=list, description="List of model configurations."
    )

    def is_dominated(self, others: List["ShapeReport"]) -> bool:
        """
        Determines whether this shape is dominated by any other shape in a Pareto sense.

        Parameters
        ----------
        others : list of ShapeReport
            List of other shape/deployment configurations to compare against.

        Returns
        -------
        bool
            True if this shape is dominated by at least one other, False otherwise.

        Notes
        -----
        A shape is dominated if there exists another configuration that is
        at least as good in all criteria and strictly better in at least one.
        Criteria:
        - Cost (to be minimized)
        - Performance, quantization level, max sequence length (to be maximized)
        """
        try:
            cand_cost = self.shape_details.gpu_specs.ranking.cost
            cand_perf = self.shape_details.gpu_specs.ranking.performance
            cand_quant = QUANT_MAPPING.get(
                self.configurations[0].deployment_params.quantization, 0
            )
            cand_maxlen = self.configurations[0].deployment_params.max_model_len

            for other in others:
                other_cost = other.shape_details.gpu_specs.ranking.cost
                other_perf = other.shape_details.gpu_specs.ranking.performance
                other_quant = QUANT_MAPPING.get(
                    other.configurations[0].deployment_params.quantization, 0
                )
                other_maxlen = other.configurations[0].deployment_params.max_model_len
                if (
                    other_cost <= cand_cost
                    and other_perf >= cand_perf
                    and other_quant >= cand_quant
                    and other_maxlen >= cand_maxlen
                    and (
                        other_cost < cand_cost
                        or other_perf > cand_perf
                        or other_quant > cand_quant
                        or other_maxlen > cand_maxlen
                    )
                ):
                    return True
            return False
        except AttributeError:
            return False

    @classmethod
    def pareto_front(cls, shapes: List["ShapeReport"]) -> List["ShapeReport"]:
        """
        Filters a list of shapes/configurations to those on the Pareto frontier.

        Parameters
        ----------
        shapes : list of ShapeReport
            List of candidate shape/configuration reports to evaluate.

        Returns
        -------
        list of ShapeReport
            Subset of input shapes that are not dominated by any other (the Pareto front).

        Notes
        -----
        The returned set contains non-dominated deployments for maximizing
        performance, quantization, and model length, while minimizing cost.
        """
        return [
            shape
            for shape in shapes
            if not shape.is_dominated([s for s in shapes if s != shape])
        ]


class ShapeRecommendationReport(BaseModel):
    """
    Full report of shape fit recommendations and troubleshooting, if applicable.

    Attributes:
        recommendations (List[DeploymentShapeSummary]): Recommended deployment shapes
            for each tested batch size and max sequence length combination.
        troubleshoot (Optional[TroubleshootShapeSummary]): Troubleshooting information
            if no valid deployment shapes are available.
    """

    display_name: Optional[str] = Field(
        "", description="Name of the model used for recommendations."
    )
    recommendations: List[ShapeReport] = Field(
        default_factory=list, description="List of shape fit recommendations."
    )
    troubleshoot: Optional[str] = Field(
        None,
        description="Details for troubleshooting if no shapes fit the current model.",
    )

    @classmethod
    def create_deployment_config_from_params_string(
        cls, config_params: str, config_env: dict
    ) -> DeploymentParams:
        """
        Parse a vLLM parameter string and create a DeploymentParams object.

        Parameters
        ----------
        config_params : str
            A space-separated string of deployment parameters
            (e.g., '--quantization mxfp4 --weight-dtype fp16 --max-model-len 4096').
            If None or empty, default parameter values are used.

        Returns
        -------
        DeploymentParams
            A DeploymentParams object populated with parsed or default values.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(QUANT_FLAG, type=str, default=None)
        parser.add_argument(
            WEIGHT_DTYPE_FLAG, dest="weight_dtype", type=str, default=None
        )
        parser.add_argument(
            MAX_MODEL_LEN_FLAG, dest="max_model_len", type=int, default=None
        )

        # Use parse_known_args to gracefully handle unexpected arguments
        args, _ = parser.parse_known_args(
            config_params.split() if config_params else []
        )

        return DeploymentParams(
            quantization=args.quantization,
            weight_dtype=args.weight_dtype,
            max_model_len=args.max_model_len,
            params=config_params or "",
            env_var=config_env,
        )

    @classmethod
    def from_deployment_config(
        cls,
        deployment_config: AquaDeploymentConfig,
        model_name: str,
        valid_shapes: List[ComputeShapeSummary],
    ) -> "ShapeRecommendationReport":
        """
        Creates a ShapeRecommendationReport from an AquaDeploymentConfig, extracting recommended
        model configurations for each valid compute shape.

        Parameters
        ----------
        deployment_config : AquaDeploymentConfig
            The object containing per-shape deployment configurations.
        model_name : str
            The name of the model for which to generate recommendations.
        valid_shapes : list of ComputeShapeSummary
            List of compute shapes to evaluate and recommend deployment configurations for.

        Returns
        -------
        ShapeRecommendationReport
            Report containing recommendations for each valid compute shape.

        Notes
        -----
        For service models, this method interprets pre-set deployment configurations to derive
        recommendations for each allowed compute shape, including environment variables, quantization,
        and maximum model length parameters.
        """

        recs = []
        for shape in valid_shapes:
            current_config = deployment_config.configuration.get(shape.name)
            if not current_config:
                continue

            recommendation = ""
            current_params = current_config.parameters.get(VLLM_PARAMS_FAMILY)
            current_env = current_config.env.get(VLLM_ENV)

            deployment_params = cls.create_deployment_config_from_params_string(
                current_params, current_env
            )

            if current_env:
                recommendation += f"ENV: {json.dumps(current_env)}\n\n"

            if (
                not current_params and not current_env
            ):  # model works with default params and no extra env variables
                recommendation += "No override PARAMS and ENV variables needed. \n\n"

            recommendation += "Model fits well within the allowed compute shape."

            # need to adjust for multiple configs per shape
            configuration = [
                ModelConfig(
                    deployment_params=deployment_params,
                    recommendation=recommendation,
                )
            ]

            recs.append(ShapeReport(shape_details=shape, configurations=configuration))

        return ShapeRecommendationReport(display_name=model_name, recommendations=recs)
