import json
import os
import re
import subprocess
import tempfile
import time
from subprocess import CompletedProcess
from typing import Optional, Dict, Union, Any, List
import webbrowser
import click
import oci.marketplace.models as models
import fsspec
import yaml
from kubernetes import config, client
from kubernetes.client import V1Pod, V1ObjectMeta
from kubernetes.stream import stream

from ads.opctl.backend.marketplace.marketplace_type import (
    HelmMarketplaceListingDetails,
    MarketplaceListingDetails,
)

from ads.opctl.backend.marketplace.marketplace_backend_runner import (
    MarketplaceBackendRunner,
)

from ads import logger

from ads.common.auth import AuthContext, AuthType
from ads.opctl.backend.marketplace.marketplace_utils import (
    export_helm_chart,
    wait_for_marketplace_export,
    list_container_images,
    get_marketplace_client,
    set_kubernetes_session_token_env,
)

from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader
from ads.opctl.operator.runtime import const as operator_runtime_const
from ads.opctl.operator.runtime import marketplace_runtime as operator_runtime
from ads.opctl.backend.base import Backend


def cleanse(prompt: str) -> str:
    return re.sub("<.*?>", "", prompt)


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

    CHECK = "\u2705"
    CROSS = "\u274C"
    LOADING = "\u274d"

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
        self.operator_config = {
            **{
                key: value
                for key, value in self.config.items()
                if key not in ("runtime", "infrastructure", "execution")
            }
        }
        self.operator_type = self.operator_config.get("type")

        self._RUNTIME_RUN_MAP = {
            operator_runtime.MarketplacePythonRuntime.type: self._run_with_python,
        }

        self.operator_info = operator_info

    def _run_helm_install(self, name, chart, **kwargs) -> CompletedProcess:
        flags = []
        for key, value in kwargs.items():
            flags.extend([f"--{key}", f"{value}"])
        helm_cmd = ["helm", "install", name, chart, *flags]
        print(" ".join(helm_cmd))
        return subprocess.run(helm_cmd)

    @staticmethod
    def _save_helm_value_to_yaml(helm_values: Dict[str, Any]) -> str:
        override_value_path = os.path.join(
            tempfile.TemporaryDirectory().name, f"values.yaml"
        )
        with fsspec.open(override_value_path, mode="w") as f:
            f.write(yaml.dump(helm_values))
        return override_value_path

    @staticmethod
    def _delete_temp_file(temp_file_path: str) -> bool:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            return True

        return False

    def _wait_for_pod_ready(self, namespace: str, pod_name: str):
        # Configs can be set in Configuration class directly or using helper utility
        # self._set_kubernetes_env()
        config.load_kube_config()
        v1 = client.CoreV1Api()

        def is_pod_ready(pod):
            for condition in pod.status.conditions:
                if condition.type == "Ready":
                    return condition.status == "True"
            return False

        start_time = time.time()
        timeout_seconds = 10 * 60
        sleep_time = 20
        while True:
            pod = v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app.kubernetes.io/instance={pod_name}",
            ).items[0]

            if is_pod_ready(pod):
                return 0
            if time.time() - start_time >= timeout_seconds:
                print("Timed out waiting for pod to get ready.")
                break
            print(f"Waiting for pod {pod_name} to be ready...")
            time.sleep(sleep_time)
        return -1

    # TODO: remove in helidon
    @staticmethod
    def run_bugfix_command(namespace: str, pod_name: str):
        # Configs can be set in Configuration class directly or using helper utility
        # self._set_kubernetes_env()
        print("Running bugfix command!!!")
        time.sleep(60)
        config.load_kube_config()
        v1 = client.CoreV1Api()

        pod: V1Pod = v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app.kubernetes.io/instance={pod_name}",
        ).items[0]
        metadata: V1ObjectMeta = pod.metadata
        resp = stream(
            v1.connect_get_namespaced_pod_exec,
            name=metadata.name,
            namespace=metadata.namespace,
            command=[
                "/bin/bash",
                "-c",
                "sed -i  's/-DuseJipherJceProvider=true//' /etc/runit/artifacts/feature-store-dataplane-api/run.sh",
            ],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        return 0

    @staticmethod
    def _export_helm_chart_to_container_registry_(
        listing_details: HelmMarketplaceListingDetails,
    ) -> Dict[str, str]:
        workflow_id = export_helm_chart(listing_details)
        wait_for_marketplace_export(workflow_id)
        images = list_container_images(listing_details)
        image_map = {}
        for image in images.items:
            for container_tag_pattern in listing_details.container_tag_pattern:
                if (
                    container_tag_pattern in image.display_name
                    and image_map.get(container_tag_pattern, None) is None
                ):
                    image_map[container_tag_pattern] = image.display_name
        return image_map

    def check_prerequisites(self, listing_details: MarketplaceListingDetails):
        compartment_id = listing_details.compartment_id
        package_version = listing_details.version
        listing_id = listing_details.listing_id
        marketplace = get_marketplace_client()
        listing: models.Listing = marketplace.get_listing(
            listing_id=listing_id, compartment_id=compartment_id
        ).data
        print("\n\n", "*" * 30, f"Checking prerequisites {self.LOADING}", "*" * 30)
        print(f"Checking license agreements for listing: {listing.name} {self.LOADING}")
        accepted_agreements: List[
            models.AcceptedAgreementSummary
        ] = marketplace.list_accepted_agreements(
            listing_id=listing_id,
            compartment_id=compartment_id,
        ).data

        agreement_summaries: List[
            models.AgreementSummary
        ] = marketplace.list_agreements(
            listing_id=listing_id,
            package_version=package_version,
            compartment_id=listing_details.compartment_id,
        ).data

        accepted_agreement_ids: List[str] = []
        for accepted_agreement in accepted_agreements:
            accepted_agreement_ids.append(accepted_agreement.agreement_id)

        for agreement_summary in agreement_summaries:
            if agreement_summary.id not in accepted_agreement_ids:
                agreement: models.Agreement = marketplace.get_agreement(
                    listing_id=listing_id,
                    package_version=package_version,
                    agreement_id=agreement_summary.id,
                    compartment_id=listing_details.compartment_id,
                ).data
                print(
                    f"Agreement from {agreement.author} with id: {agreement_summary.id} is not accepted. Opening terms and conditions in default browser {self.LOADING}"
                )
                webbrowser.open(agreement.content_url)
                time.sleep(1)
                answer = click.confirm(
                    f"{cleanse(agreement.prompt)}",
                    default=False,
                )
                if not answer:
                    print(
                        "*" * 30,
                        f"Agreement from author: {agreement_summary.author} with id: {agreement_summary.id} "
                        f"is rejected {self.CROSS} "
                        "*" * 30,
                    )
                    raise Exception
                else:
                    create_accepted_agreement_details: models.CreateAcceptedAgreementDetails = (
                        models.CreateAcceptedAgreementDetails()
                    )
                    create_accepted_agreement_details.agreement_id = (
                        agreement_summary.id
                    )
                    create_accepted_agreement_details.listing_id = listing_id
                    create_accepted_agreement_details.compartment_id = compartment_id
                    create_accepted_agreement_details.signature = agreement.signature
                    create_accepted_agreement_details.package_version = package_version
                    marketplace.create_accepted_agreement(
                        create_accepted_agreement_details
                    )

            print(
                f"Agreement from author: {agreement_summary.author} with id: {agreement_summary.id} is accepted "
                f"{self.CHECK} "
            )

    def _run_with_python(self, **kwargs: Dict) -> int:
        """
        Runs the operator within a local python environment.

        Returns
        -------
        int
            The operator's run exit code.
        """

        # build runtime object
        with AuthContext(auth=self.auth_type, profile=self.profile):
            if self.auth_type == AuthType.SECURITY_TOKEN:
                set_kubernetes_session_token_env()
            runtime = operator_runtime.MarketplacePythonRuntime.from_dict(
                self.runtime_config, ignore_unknown=True
            )

            # run operator
            operator_spec = json.dumps(self.operator_config)
            operator = MarketplaceBackendRunner(
                module_name=self.operator_info.type,
            )

            listing_details: MarketplaceListingDetails = operator.get_listing_details(
                operator_spec
            )
            self.check_prerequisites(listing_details)
            if isinstance(listing_details, HelmMarketplaceListingDetails):
                listing_details: HelmMarketplaceListingDetails = listing_details
                container_map = self._export_helm_chart_to_container_registry_(
                    listing_details
                )
                oci_meta = operator.get_oci_meta(container_map, operator_spec)
                listing_details.helm_values["oci_meta"] = oci_meta
                override_value_path = self._save_helm_value_to_yaml(
                    listing_details.helm_values
                )
                helm_install_status = self._run_helm_install(
                    name=listing_details.helm_app_name,
                    chart="oci://" + listing_details.ocir_repo.rstrip("/"),
                    # + "/"
                    **{
                        # TODO: Fix after feature store listing,
                        # "version": listing_details.version,
                        "version": listing_details.helm_chart_tag,
                        "namespace": listing_details.namespace,
                        "values": override_value_path,
                    },
                )
                self.run_bugfix_command(
                    namespace=listing_details.namespace,
                    pod_name=listing_details.helm_app_name,
                )
                if helm_install_status.returncode == 0:
                    return self._wait_for_pod_ready(
                        listing_details.namespace, listing_details.helm_app_name
                    )
                else:
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
