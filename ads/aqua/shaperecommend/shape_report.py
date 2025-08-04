#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional

from pydantic import BaseModel, Field

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.shaperecommend.constants import QUANT_MAPPING
from ads.aqua.shaperecommend.estimator import MemoryEstimator
from ads.config import COMPARTMENT_OCID


class RequestRecommend(BaseModel):
    """
    A request to recommend compute shapes and parameters for a given model.
    """

    model_id: str = Field(
        ..., description="The OCID of the model to recommend feasible compute shapes."
    )
    shapes: List[ComputeShapeSummary] = Field(
        ..., description="The list of shapes on OCI."
    )
    generate_table : Optional[bool] = Field(
        True, description="True - to generate the rich diff Table, False - generate the JSON response"
    ),
    compartment_id: Optional[str] = Field(
        COMPARTMENT_OCID, description="The OCID of user's compartment"
    )

    class Config:
        protected_namespaces = ()


class DeploymentParams(BaseModel):  # noqa: N801
    """
    Recommended parameters for deployment and model inferencing (specific to compute shape & model).
    """

    quantization: Optional[str] = Field(
        None, description="Type of quantization (e.g. 4bit)."
    )
    max_model_len: int = Field(..., description="Maximum length of input sequence.")
    params: str = Field(
        ..., description="Runtime parameters for deployment with vLLM, etc."
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

    model_details: ModelDetail = Field(..., description="Details about the model.")
    deployment_params: DeploymentParams = Field(
        ..., description="Parameters for deployment."
    )
    recommendation: str = Field(..., description="GPU recommendation for the model.")

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
        estimator : MemoryEstimator
            Estimator with model details and processed config.
        allowed_gpu_memory : float
            Maximum allowed GPU memory (in GBs) for this configuration.

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
