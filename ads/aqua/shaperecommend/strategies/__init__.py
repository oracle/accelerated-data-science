#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Strategy pattern for architecture-specific shape recommendation.

Each strategy encapsulates the logic needed to recommend GPU shapes
for a particular model architecture (text-generation, multimodal,
embedding, audio).
"""

from ads.aqua.shaperecommend.strategies.audio import AudioStrategy
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy
from ads.aqua.shaperecommend.strategies.embedding import EmbeddingStrategy
from ads.aqua.shaperecommend.strategies.multimodal import MultimodalStrategy
from ads.aqua.shaperecommend.strategies.text import TextGenerationStrategy

__all__ = [
    "RecommendationStrategy",
    "TextGenerationStrategy",
    "MultimodalStrategy",
    "EmbeddingStrategy",
    "AudioStrategy",
]
