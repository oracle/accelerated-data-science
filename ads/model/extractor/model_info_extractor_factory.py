#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.model_metadata import Framework
from ads.common import logger, utils
from ads.model.extractor.sklearn_extractor import SklearnExtractor
from ads.model.extractor.xgboost_extractor import XgboostExtractor
from ads.model.extractor.lightgbm_extractor import LightgbmExtractor
from ads.model.extractor.keras_extractor import KerasExtractor
from ads.model.extractor.automl_extractor import AutoMLExtractor
from ads.model.extractor.spark_extractor import SparkExtractor
from ads.model.extractor.pytorch_extractor import PytorchExtractor
from ads.model.extractor.tensorflow_extractor import TensorflowExtractor
from ads.model.extractor.huggingface_extractor import HuggingFaceExtractor


ORDERED_FRAMEWORKS = [
    "lightgbm",
    "xgboost",
    "sklearn",
    "keras",
    "tensorflow",
    "bert",
    "transformers",
    "torch",
    "spark",
    "automl",
]


class ModelInfoExtractorFactory:
    """Class that extract Model Taxonomy Metadata for all supported frameworks."""

    _estimator_map = {
        Framework.SCIKIT_LEARN: SklearnExtractor,
        Framework.XGBOOST: XgboostExtractor,
        Framework.LIGHT_GBM: LightgbmExtractor,
        Framework.KERAS: KerasExtractor,
        Framework.ORACLE_AUTOML: AutoMLExtractor,
        Framework.TENSORFLOW: TensorflowExtractor,
        Framework.PYTORCH: PytorchExtractor,
        Framework.SPARK: SparkExtractor,
        Framework.TRANSFORMERS: HuggingFaceExtractor,
    }

    @staticmethod
    def extract_info(model):
        """Extracts model taxonomy metadata.


        Parameters
        ----------
        model: [ADS model, sklearn, xgboost, lightgbm, keras, oracle_automl]
            The model object

        Returns
        -------
        `ModelTaxonomyMetadata`
            A dictionary with keys of Framework, FrameworkVersion, Algorithm, Hyperparameters of the model

        Examples
        --------
        >>> from ads.common.model_info_extractor_factory import ModelInfoExtractorFactory
        >>> metadata_taxonomy = ModelInfoExtractorFactory.extract_info(model)

        """
        from ads.common.model import ADSModel

        if isinstance(model, ADSModel):
            model = model.est
        model_framework = None
        model_bases = utils.get_base_modules(model)
        model_framework = ModelInfoExtractorFactory._get_estimator(
            model_bases=model_bases
        )
        if model_framework not in ModelInfoExtractorFactory._estimator_map:
            logger.warn(
                f"Auto-extraction of taxonomy is not supported for the provided model. "
                f"The supported models are {', '.join(ORDERED_FRAMEWORKS)}."
            )
            return None
        return ModelInfoExtractorFactory._estimator_map[model_framework](model).info()

    @staticmethod
    def _get_estimator(model_bases):
        mapping = {
            "lightgbm": Framework.LIGHT_GBM,
            "xgboost": Framework.XGBOOST,
            "sklearn": Framework.SCIKIT_LEARN,
            "keras": Framework.KERAS,
            "tensorflow": Framework.TENSORFLOW,
            "bert": Framework.BERT,
            "transformers": Framework.TRANSFORMERS,
            "torch": Framework.PYTORCH,
            "spark": Framework.SPARK,
            "automl": Framework.ORACLE_AUTOML,
        }
        for model_base in model_bases:
            for framework in ORDERED_FRAMEWORKS:
                if framework in model_base.__module__.split("."):
                    return mapping.get(framework)
        return None
