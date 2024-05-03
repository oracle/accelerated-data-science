#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import tempfile
from typing import Optional, Dict, Union, Any
import fsspec
import yaml
from ads.opctl.backend.marketplace.prerequisite_checker import check_prerequisites

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.opctl.backend.marketplace.helm_helper import check_helm_login, run_helm_install
from ads.opctl.backend.marketplace.marketplace_operator_interface import Status

from ads.opctl.backend.marketplace.models.marketplace_type import (
    HelmMarketplaceListingDetails,
    MarketplaceListingDetails,
)

from ads.opctl.backend.marketplace.marketplace_backend_runner import (
    MarketplaceBackendRunner,
)

from ads import logger

from ads.common.auth import AuthContext, AuthType
from ads.opctl.backend.marketplace.marketplace_utils import (
    set_kubernetes_session_token_env,
    StatusIcons,
    print_heading,
    Color,
    export_if_tags_not_exist,
    get_kubernetes_service,
)

from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader
from ads.opctl.operator.runtime import const as operator_runtime_const
from ads.opctl.operator.runtime import marketplace_runtime as operator_runtime
from ads.opctl.backend.base import Backend


class LocalMarketplaceOperatorBackend(Backend):
    """
    The local operator backend to execute operator in the local environment.
    Currently supported two scenarios:
        * Running operator within local conda environment.
        * Running operator within local container.

    Attributes
    ----------
    runtime_config: (Dict)
        The runtime config for the operator.
    operator_config: (Dict)
        The operator specification config.
    operator_type: str
        The type of the operator.
    operator_info: OperatorInfo
        The detailed information about the operator.
    """

    def __init__(
        self, config: Optional[Dict], operator_info: OperatorInfo = None
    ) -> None:
        """
        Instantiates the operator backend.

        Parameters
        ----------
        config: (Dict)
            The configuration file containing operator's specification details and execution section.
        operator_info: (OperatorInfo, optional)
            The operator's detailed information extracted from the operator.__init__ file.
            Will be extracted from the operator type in case if not provided.
        """
        super().__init__(config=config or {})
        self.runtime_config = self.config.get("runtime", {})
        self.spec = self.runtime_config.get("spec", {})
        self.operator_config = {
            **{
                key: value
                for key, value in self.config.items()
                if key not in ("runtime", "infrastructure", "execution")
            }
        }
        self.operator_type = self.operator_config.get("type")

        self._RUNTIME_RUN_MAP = {
            operator_runtime.MarketplacePythonRuntime.type: self._run_with_python_,
        }
        self.operator = None
        self.operator_info = operator_info

    @staticmethod
    def _save_helm_values_to_yaml_(helm_values: Dict[str, Any]) -> str:
        temp_file_path = os.path.join(
            tempfile.TemporaryDirectory().name, f"values.yaml"
        )
        with fsspec.open(temp_file_path, mode="w") as f:
            f.write(yaml.dump(helm_values))
        return temp_file_path

    def process_helm_listing(self, listing_details: HelmMarketplaceListingDetails):
        operator = MarketplaceBackendRunner(
            module_name=self.operator_info.type,
        )
        check_prerequisites(listing_details)
        print_heading(
            f"Starting deployment",
            prefix_newline_count=2,
            suffix_newline_count=0,
            colors=[Color.BLUE, Color.BOLD],
        )

        tags_map = export_if_tags_not_exist(listing_details)
        check_helm_login(listing_details)
        oci_meta = operator.get_oci_meta(json.dumps(self.operator_config), tags_map)
        listing_details.helm_values["oci_meta"] = oci_meta
        override_value_path = self._save_helm_values_to_yaml_(
            listing_details.helm_values
        )
        helm_install_status = run_helm_install(
            name=listing_details.helm_app_name,
            chart=listing_details.helm_fully_qualified_url,
            version=listing_details.helm_chart_tag,
            namespace=listing_details.namespace,
            values_yaml_path=override_value_path,
        )
        if helm_install_status.returncode == 0:
            print_heading(
                f"Completed deployment {StatusIcons.TADA}",
                colors=[Color.BOLD, Color.BLUE],
                prefix_newline_count=0,
                suffix_newline_count=2,
            )

            operator.finalise_installation(
                json.dumps(self.operator_config),
                Status.SUCCESS,
                tags_map,
                get_kubernetes_service(listing_details),
            )
        else:
            operator.finalise_installation(
                json.dumps(self.operator_config), Status.FAILURE, None, None
            )
        return helm_install_status.returncode

    @runtime_dependency(
        module="kubernetes", install_from=OptionalDependency.FEATURE_STORE_MARKETPLACE
    )
    def _run_with_python_(self, **kwargs: Dict) -> int:
        """
        Runs the operator within a local python environment.

        Returns
        -------
        int
            The operator's run exit code.
        """
        self.operator = MarketplaceBackendRunner(module_name=self.operator_info.type)

        # build runtime object
        with AuthContext(auth=self.auth_type, profile=self.profile):
            if self.auth_type == AuthType.SECURITY_TOKEN:
                set_kubernetes_session_token_env(profile=self.profile)
            listing_details: MarketplaceListingDetails = (
                self.operator.get_listing_details(json.dumps(self.operator_config))
            )
            if isinstance(listing_details, HelmMarketplaceListingDetails):
                return self.process_helm_listing(listing_details)
            else:
                print_heading(
                    f"This type of listing is not supported currently via marketplace backend. "
                    f"Currently only listings of type 'Helm charts' is supported",
                    colors=[Color.BOLD, Color.RED],
                )
                return -1

    def run(self, **kwargs: Dict) -> None:
        """Runs the operator."""

        # extract runtime
        runtime_type = self.runtime_config.get(
            "type", operator_runtime.OPERATOR_MARKETPLACE_LOCAL_RUNTIME_TYPE.PYTHON
        )

        if runtime_type not in self._RUNTIME_RUN_MAP:
            raise RuntimeError(
                f"Not supported runtime - {runtime_type} for local backend. "
                f"Supported values: {self._RUNTIME_RUN_MAP.keys()}"
            )
        if not self.operator_info:
            self.operator_info = OperatorLoader.from_uri(self.operator_type).load()

        if self.config.get("dry_run"):
            logger.info(
                "The dry run option is not supported for "
                "the local backends and will be ignored."
            )

        # run operator with provided runtime
        exit_code = self._RUNTIME_RUN_MAP.get(runtime_type, lambda: None)()

        if exit_code != 0:
            raise RuntimeError(
                f"Operation did not complete successfully. Exit code: {exit_code}. "
                f"Run with the --debug argument to view logs."
            )

    def init(
        self,
        uri: Union[str, None] = None,
        overwrite: bool = False,
        runtime_type: Union[str, None] = None,
        **kwargs: Dict,
    ) -> Union[str, None]:
        """Generates a starter YAML specification for the operator local runtime.

        Parameters
        ----------
        overwrite: (bool, optional). Defaults to False.
            Overwrites the result specification YAML if exists.
        uri: (str, optional), Defaults to None.
            The filename to save the resulting specification template YAML.
        runtime_type: (str, optional). Defaults to None.
                The resource runtime type.
        **kwargs: Dict
            The optional arguments.

        Returns
        -------
        Union[str, None]
            The YAML specification for the given resource if `uri` was not provided.
            `None` otherwise.
        """
        runtime_type = runtime_type or operator_runtime.MarketplacePythonRuntime.type
        if runtime_type not in operator_runtime_const.MARKETPLACE_RUNTIME_MAP:
            raise ValueError(
                f"Not supported runtime type {runtime_type}. "
                f"Supported values: {operator_runtime_const.MARKETPLACE_RUNTIME_MAP.keys()}"
            )

        runtime_kwargs_map = {
            operator_runtime.MarketplacePythonRuntime.type: {},
        }

        with AuthContext(auth=self.auth_type, profile=self.profile):
            note = (
                "# This YAML specification was auto generated by the "
                "`ads operator init` command.\n"
                "# The more details about the operator's runtime YAML "
                "specification can be found in the ADS documentation:\n"
                "# https://accelerated-data-science.readthedocs.io/en/latest \n\n"
            )

            return (
                operator_runtime_const.MARKETPLACE_RUNTIME_MAP[runtime_type]
                .init(**runtime_kwargs_map[runtime_type])
                .to_yaml(
                    uri=uri,
                    overwrite=overwrite,
                    note=note,
                    **kwargs,
                )
            )
