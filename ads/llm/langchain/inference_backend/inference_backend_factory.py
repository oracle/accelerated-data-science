#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Optional, Type

import tabulate

from ads.llm.langchain.inference_backend.const import InferenceFramework
from ads.llm.langchain.inference_backend.errors import (
    UnsupportedInferenceFrameworkError,
)
from ads.llm.langchain.inference_backend.inference_backend import (
    InferenceBackend,
    InferenceBackendGeneric,
    InferenceBackendLLamaCPP,
    InferenceBackendTGI,
    InferenceBackendVLLM,
)

logger = logging.getLogger(__name__)


class InferenceBackendFactory:
    """
    Factory class to create instances of different inference backends based on the framework.
    """

    _backend_registry = {
        InferenceFramework.GENERIC: InferenceBackendGeneric,
        InferenceFramework.VLLM: InferenceBackendVLLM,
        InferenceFramework.TGI: InferenceBackendTGI,
        InferenceFramework.LLAMA_CPP: InferenceBackendLLamaCPP,
    }

    @staticmethod
    def get_backend(framework: Optional[str] = None) -> Type[InferenceBackend]:
        """
        Returns the appropriate InferenceBackend class based on the provided framework.

        Parameters
        ----------
        framework (Optional[str]): The framework for which to get the backend.
                                    Defaults to InferenceFramework.GENERIC.

        Returns
        -------
        Type[InferenceBackend]: The class of the requested InferenceBackend.

        Raises
        ------
        UnsupportedInferenceFrameworkError: If the framework is not supported.
        """
        framework = framework or InferenceFramework.GENERIC

        if framework not in InferenceBackendFactory._backend_registry:
            raise UnsupportedInferenceFrameworkError(framework)

        return InferenceBackendFactory._backend_registry[framework]

    @staticmethod
    def supported_frameworks() -> List[str]:
        """
        Returns a list of all supported inference frameworks.

        Returns
        -------
        List[str]: A list of supported inference framework names.
        """
        return InferenceFramework.values()

    @staticmethod
    def help(framework: Optional[str] = None) -> None:
        help_data = []
        frameworks = [framework] if framework else InferenceFramework.values()

        for framework in frameworks:
            backend = InferenceBackendFactory.get_backend(framework)
            params = tabulate.tabulate(
                list(backend.ModelParams().dict().items()),
                headers=("Param", "Value"),
                tablefmt="fancy_grid",
            )
            help_data.append((backend.TYPE, f"{backend.DESCRIPTION}\n{params}"))

        print(
            tabulate.tabulate(
                help_data, headers=("Framework", "Description"), tablefmt="simple_grid"
            )
        )
