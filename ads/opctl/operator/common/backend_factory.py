#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module contains the factory method to create the backend object for the operator.
The factory validates the backend type and runtime type before creating the backend object.
"""

from typing import Dict, List, Tuple, Union

import yaml
from ads.opctl.operator.common.utils import print_traceback

from ads.opctl.backend.marketplace.local_marketplace import (
    LocalMarketplaceOperatorBackend,
)

from ads.opctl import logger
from ads.opctl.backend.ads_dataflow import DataFlowOperatorBackend
from ads.opctl.backend.ads_ml_job import MLJobOperatorBackend
from ads.opctl.backend.base import Backend
from ads.opctl.backend.local import (
    LocalOperatorBackend,
)
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import (
    BACKEND_NAME,
    DEFAULT_ADS_CONFIG_FOLDER,
    OVERRIDE_KWARGS,
    RESOURCE_TYPE,
    RUNTIME_TYPE,
)
from ads.opctl.operator.common.const import PACK_TYPE, OPERATOR_BACKEND_SECTION_NAME
from ads.opctl.operator.common.dictionary_merger import DictionaryMerger
from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader


class BackendFactory:
    """
    Class which contains the factory method to create the backend object.
    The operator's backend object is created based on the backend type.
    """

    BACKENDS = (
        BACKEND_NAME.JOB.value,
        BACKEND_NAME.DATAFLOW.value,
        BACKEND_NAME.MARKETPLACE.value,
    )

    LOCAL_BACKENDS = (
        BACKEND_NAME.OPERATOR_LOCAL.value,
        BACKEND_NAME.LOCAL.value,
    )

    BACKEND_RUNTIME_MAP = {
        BACKEND_NAME.JOB.value.lower(): {
            RUNTIME_TYPE.PYTHON.value.lower(): (
                BACKEND_NAME.JOB.value.lower(),
                RUNTIME_TYPE.PYTHON.value.lower(),
            ),
            RUNTIME_TYPE.CONTAINER.value.lower(): (
                BACKEND_NAME.JOB.value.lower(),
                RUNTIME_TYPE.CONTAINER.value.lower(),
            ),
        },
        BACKEND_NAME.DATAFLOW.value.lower(): {
            RUNTIME_TYPE.DATAFLOW.value.lower(): (
                BACKEND_NAME.DATAFLOW.value.lower(),
                RUNTIME_TYPE.DATAFLOW.value.lower(),
            )
        },
        BACKEND_NAME.OPERATOR_LOCAL.value.lower(): {
            RUNTIME_TYPE.PYTHON.value.lower(): (
                BACKEND_NAME.OPERATOR_LOCAL.value.lower(),
                RUNTIME_TYPE.PYTHON.value.lower(),
            ),
            RUNTIME_TYPE.CONTAINER.value.lower(): (
                BACKEND_NAME.OPERATOR_LOCAL.value.lower(),
                RUNTIME_TYPE.CONTAINER.value.lower(),
            ),
        },
        BACKEND_NAME.MARKETPLACE.value.lower(): {
            RUNTIME_TYPE.PYTHON.value.lower(): (
                BACKEND_NAME.MARKETPLACE.value.lower(),
                RUNTIME_TYPE.PYTHON.value.lower(),
            )
        },
    }

    BACKEND_MAP = {
        BACKEND_NAME.JOB.value.lower(): MLJobOperatorBackend,
        BACKEND_NAME.DATAFLOW.value.lower(): DataFlowOperatorBackend,
        BACKEND_NAME.OPERATOR_LOCAL.value.lower(): LocalOperatorBackend,
        BACKEND_NAME.LOCAL.value.lower(): LocalOperatorBackend,
        BACKEND_NAME.MARKETPLACE.value.lower(): LocalMarketplaceOperatorBackend,
    }

    @classmethod
    def backend(
        cls, config: ConfigProcessor, backend: Union[Dict, str] = None, **kwargs: Dict
    ) -> Backend:
        """
        The factory method to create the backend object.

        Parameters
        ----------
        config: ConfigProcessor
            The config processor object.
        backend: (Union[Dict, str], optional)
            The backend type. Can be a string or a dictionary.
        **kwargs: Dict
            The keyword arguments.

        Returns
        -------
        Returns the backend object.

        Raises
        ------
        RuntimeError
            If the backend type is not supported.
        """
        if not config:
            raise RuntimeError("The config is not provided.")

        if config.config.get("kind", "").lower() != "operator":
            raise RuntimeError("Not supported kind of workload.")

        operator_type = config.config.get("type", "").lower()

        # validation
        if not operator_type:
            raise RuntimeError(
                f"The `type` attribute must be specified in the operator's config."
            )

        if not backend and not config.config.get(OPERATOR_BACKEND_SECTION_NAME):
            logger.info(
                f"Backend config is not provided, the {BACKEND_NAME.LOCAL.value} "
                "will be used by default. "
            )
            backend = BACKEND_NAME.LOCAL.value
        elif not backend:
            backend = config.config.get(OPERATOR_BACKEND_SECTION_NAME)

        # extracting details about the operator
        operator_info = OperatorLoader.from_uri(uri=operator_type).load()

        supported_backends = tuple(
            set(cls.BACKENDS + cls.LOCAL_BACKENDS) & set(operator_info.backends)
        )

        runtime_type = None
        backend_kind = None

        if isinstance(backend, str):
            backend_kind, runtime_type = cls._extract_backend(
                backend=backend, supported_backends=supported_backends
            )
            backend = {"kind": backend_kind}

        backend_kind = (
            BACKEND_NAME.OPERATOR_LOCAL.value
            if backend.get("kind").lower() == BACKEND_NAME.LOCAL.value
            else backend.get("kind").lower()
        )
        backend["kind"] = backend_kind

        # If the backend kind is Job, then it is necessary to check the infrastructure kind.
        # This is necessary, because Jobs and DataFlow have similar kind,
        # The only difference would be in the infrastructure kind.
        # This is a temporary solution, the logic needs to be placed in the ConfigMerger instead.
        if backend_kind == BACKEND_NAME.JOB.value:
            if (backend.get("spec", {}) or {}).get("infrastructure", {}).get(
                "type", ""
            ).lower() == BACKEND_NAME.DATAFLOW.value:
                backend_kind = BACKEND_NAME.DATAFLOW.value

        runtime_type = runtime_type or (
            backend.get("type")
            or (backend.get("spec", {}) or {})
            .get("runtime", {})
            .get("type", "undefined")
        )

        # validation
        cls._validate_backend_and_runtime(
            backend_kind=backend_kind,
            runtime_type=runtime_type,
            supported_backends=supported_backends,
        )

        # generate backend specification in case if it is not provided
        if not backend.get("spec"):
            backends = cls._init_backend_config(
                operator_info=operator_info, backend_kind=backend_kind, **kwargs
            )

            backend = backends.get(cls.BACKEND_RUNTIME_MAP[backend_kind][runtime_type])
            if not backend:
                raise RuntimeError(
                    "An error occurred while attempting to load the "
                    f"configuration for the `{backend_kind}.{runtime_type}` backend."
                )

        p_backend = ConfigProcessor(
            {**backend, **{"execution": {"backend": backend_kind}}}
        ).step(ConfigMerger, **kwargs)

        # merge backend with the override parameters
        config.config["runtime"] = DictionaryMerger(
            updates=kwargs.get(OVERRIDE_KWARGS)
        ).merge(backend)
        config.config["infrastructure"] = p_backend.config["infrastructure"]
        config.config["execution"] = p_backend.config["execution"]

        return cls.BACKEND_MAP[
            p_backend.config["execution"][OPERATOR_BACKEND_SECTION_NAME].lower()
        ](config=config.config, operator_info=operator_info)

    @classmethod
    def _extract_backend(
        cls, backend: str, supported_backends: List[str] = None
    ) -> Tuple[str, str]:
        """
        Extracts the backend type and the runtime type from the backend string.

        Parameters
        ----------
        backend: str
            The backend string.
            Example: `job`, `job.container`, `dataflow.dataflow`, `local.container`, `local.python`.
        supported_backends: List[str]
            The list of supported backends.

        Returns
        -------
        Returns the tuple of the backend type and the runtime type.

        Raises
        ------
        RuntimeError
            If the backend type is not supported.
        """
        supported_backends = supported_backends or (cls.BACKENDS + cls.LOCAL_BACKENDS)
        backend = (backend or BACKEND_NAME.OPERATOR_LOCAL.value).lower()
        backend_kind, runtime_type = backend, None

        if backend.lower() != BACKEND_NAME.OPERATOR_LOCAL.value and "." in backend:
            backend_kind, runtime_type = backend.split(".")
        else:
            backend_kind = backend

        backend_kind = (
            BACKEND_NAME.OPERATOR_LOCAL.value
            if backend_kind == BACKEND_NAME.LOCAL.value
            else backend_kind
        )

        if backend_kind not in supported_backends:
            raise RuntimeError(
                f"Not supported backend - {backend_kind}. Supported backends: {supported_backends}"
            )

        runtime_type = (
            runtime_type or list(cls.BACKEND_RUNTIME_MAP[backend_kind].keys())[0]
        )

        if runtime_type not in cls.BACKEND_RUNTIME_MAP[backend_kind]:
            raise RuntimeError(
                f"Not supported runtime type - `{runtime_type}` for the backend - `{backend_kind}`. "
                f"Supported runtime types: `{list(cls.BACKEND_RUNTIME_MAP[backend_kind].keys())}`"
            )

        return backend_kind, runtime_type

    @classmethod
    def _validate_backend_and_runtime(
        cls, backend_kind: str, runtime_type: str, supported_backends: List[str] = None
    ) -> bool:
        """
        Validates the backend kind and runtime type.

        Parameters
        ----------
        backend_kind: str
            The backend kind.
        runtime_type: str
            The runtime type.
        supported_backends: List[str]
            The list of supported backends.

        Returns
        -------
        Returns True if the backend type is valid, otherwise False.

        Raises
        ------
        RuntimeError
            If the backend type is not supported.
        """
        supported_backends = supported_backends or (cls.BACKENDS + cls.LOCAL_BACKENDS)
        if backend_kind not in supported_backends:
            raise RuntimeError(
                f"Not supported backend - {backend_kind}. Supported backends: {supported_backends}"
            )
        if runtime_type not in cls.BACKEND_RUNTIME_MAP[backend_kind]:
            raise RuntimeError(
                f"Not supported runtime type - `{runtime_type}` for the backend - `{backend_kind}`. "
                f"Supported runtime types: `{list(cls.BACKEND_RUNTIME_MAP[backend_kind].keys())}`"
            )
        return True

    @classmethod
    def _init_backend_config(
        cls,
        operator_info: OperatorInfo,
        ads_config: Union[str, None] = None,
        backend_kind: Tuple[str] = None,
        **kwargs: Dict,
    ) -> Dict[Tuple, Dict]:
        """
        Generates the operator's backend configs.

        Parameters
        ----------
        ads_config: (str, optional)
            The folder where the ads opctl config located.
        backend_kind: (str, optional)
            The required backend.
        kwargs: (Dict, optional).
            Additional key value arguments.

        Returns
        -------
        Dict[Tuple, Dict]
            The dictionary where the key will be a tuple containing runtime kind and type.
            Example:
            >>> {("local","python"): {}, ("job", "container"): {}}

        Raises
        ------
        RuntimeError
            In case if the provided backend is not supported.
        """
        from ads.opctl.cmds import _BackendFactory

        result = {}

        freeform_tags = {
            "operator": f"{operator_info.type}:{operator_info.version}",
        }

        # generate supported backend specifications templates YAML
        RUNTIME_TYPE_MAP = {
            RESOURCE_TYPE.JOB.value: [
                {
                    RUNTIME_TYPE.PYTHON: {
                        "conda_slug": operator_info.conda
                        if operator_info.conda_type == PACK_TYPE.SERVICE
                        else operator_info.conda_prefix,
                        "freeform_tags": freeform_tags,
                    }
                },
                {
                    RUNTIME_TYPE.CONTAINER: {
                        "image_name": f"{operator_info.type}:{operator_info.version}",
                        "freeform_tags": freeform_tags,
                    }
                },
            ],
            RESOURCE_TYPE.DATAFLOW.value: [
                {
                    RUNTIME_TYPE.DATAFLOW: {
                        "conda_slug": operator_info.conda_prefix,
                        "freeform_tags": freeform_tags,
                    }
                }
            ],
            BACKEND_NAME.OPERATOR_LOCAL.value: [
                {
                    RUNTIME_TYPE.CONTAINER: {
                        "kind": "operator",
                        "type": operator_info.type,
                        "version": operator_info.version,
                    }
                },
                {
                    RUNTIME_TYPE.PYTHON: {
                        "kind": "operator",
                        "type": operator_info.type,
                        "version": operator_info.version,
                    }
                },
            ],
            BACKEND_NAME.MARKETPLACE.value: [
                {
                    RUNTIME_TYPE.PYTHON: {
                        "kind": "marketplace",
                        "type": operator_info.type,
                        "version": operator_info.version,
                    }
                }
            ],
        }

        supported_backends = tuple(
            set(RUNTIME_TYPE_MAP.keys()) & set(operator_info.backends)
        )

        if backend_kind:
            if backend_kind not in supported_backends:
                raise RuntimeError(
                    f"Not supported backend - {backend_kind}. Supported backends: {supported_backends}"
                )
            supported_backends = (backend_kind,)

        for resource_type in supported_backends:
            try:
                for runtime_type_item in RUNTIME_TYPE_MAP.get(
                    resource_type.lower(), []
                ):
                    runtime_type, runtime_kwargs = next(iter(runtime_type_item.items()))

                    # get config info from ini files
                    p = ConfigProcessor(
                        {
                            **runtime_kwargs,
                            **{"execution": {"backend": resource_type}},
                            **{
                                "infrastructure": {
                                    **operator_info.jobs_default_params.to_dict(),
                                    **operator_info.dataflow_default_params.to_dict(),
                                }
                            },
                        }
                    ).step(
                        ConfigMerger,
                        ads_config=ads_config or DEFAULT_ADS_CONFIG_FOLDER,
                        **kwargs,
                    )

                    # generate YAML specification template
                    result[
                        (resource_type.lower(), runtime_type.value.lower())
                    ] = yaml.load(
                        _BackendFactory(p.config).backend.init(
                            runtime_type=runtime_type.value,
                            **{**kwargs, **runtime_kwargs},
                        ),
                        Loader=yaml.FullLoader,
                    )
            except Exception as ex:
                logger.warning(
                    f"Unable to generate the configuration for the `{resource_type}` backend. "
                    f"{ex}"
                )
                print_traceback()

        return result
