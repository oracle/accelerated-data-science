#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Callable, Dict, Self

from ads.model.extractor.embedding_onnx_extractor import EmbeddingONNXExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.common import SERDE


class EmbeddingONNXModel(FrameworkSpecificModel):
    def __init__(
        self,
        estimator: Callable[..., Any] = None,
        artifact_dir: str | None = None,
        properties: ModelProperties | None = None,
        auth: Dict | None = None,
        serialize: bool = True,
        model_save_serializer: SERDE | None = None,
        model_input_serializer: SERDE | None = None,
        **kwargs: dict,
    ) -> Self:
        super().__init__(
            estimator,
            artifact_dir,
            properties,
            auth,
            serialize,
            model_save_serializer,
            model_input_serializer,
            **kwargs,
        )

        self._extractor = EmbeddingONNXExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter
