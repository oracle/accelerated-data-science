#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.shaperecommend.shape_report import ShapeRecommendationReport
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig


class RecommendationStrategy(ABC):
    """
    Abstract base class for architecture-specific shape recommendation strategies.
    
    Each strategy handles a specific model architecture type (text-generation,
    multimodal, embedding, audio) and encapsulates the logic for:
    - Creating the appropriate memory estimator
    - Determining which shapes are compatible
    - Building deployment parameters (vLLM flags, env vars)
    """

    @abstractmethod
    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """
        Generates shape recommendations for the given model configuration.
        
        Parameters
        ----------
        parsed_config : ParsedModelConfig
            The parsed model configuration with architecture-specific sub-configs.
        shapes : List[ComputeShapeSummary]
            List of available compute shapes, sorted by GPU memory descending.
        model_name : str
            Display name of the model.
        batch_size : int, optional
            Batch size for estimation (default 1).
            
        Returns
        -------
        ShapeRecommendationReport
            The recommendation report with compatible shapes or troubleshooting info.
        """
        pass
