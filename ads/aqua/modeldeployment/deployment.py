#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import shlex
from typing import Dict, List, Optional, Union

from ads.aqua.app import AquaApp, logger
from ads.aqua.common.entities import ContainerSpec
from ads.aqua.common.enums import (
    InferenceContainerTypeFamily,
    Tags,
)
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.aqua.common.utils import (
    get_combined_params,
    get_container_config,
    get_container_image,
    get_container_params_type,
    get_model_by_reference_paths,
    get_ocid_substring,
    get_params_dict,
    get_params_list,
    get_resource_name,
    get_restricted_params_by_container,
    validate_cmd_var,
)
from ads.aqua.constants import (
    AQUA_MODEL_ARTIFACT_FILE,
    AQUA_MODEL_TYPE_CUSTOM,
    AQUA_MODEL_TYPE_SERVICE,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    UNKNOWN,
    UNKNOWN_DICT,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.finetuning.finetuning import FineTuneCustomMetadata
from ads.aqua.model import AquaModelApp
from ads.aqua.modeldeployment.entities import (
    AquaDeployment,
    AquaDeploymentDetail,
)
from ads.aqua.ui import ModelFormat
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import get_log_links
from ads.config import (
    AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME,
    AQUA_MODEL_DEPLOYMENT_CONFIG,
    COMPARTMENT_OCID,
)
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment import (
    ModelDeployment,
    ModelDeploymentContainerRuntime,
    ModelDeploymentInfrastructure,
    ModelDeploymentMode,
)
from ads.telemetry import telemetry


class AquaDeploymentApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua model deployments within the Oracle
    Cloud Infrastructure Data Science service, serving as an interface for deploying
    machine learning models.


    Methods
    -------
    create(model_id: str, instance_shape: str, display_name: str,...) -> AquaDeployment
        Creates a model deployment for Aqua Model.
    get(model_deployment_id: str) -> AquaDeployment:
        Retrieves details of an Aqua model deployment by its unique identifier.
    list(**kwargs) -> List[AquaModelSummary]:
        Lists all Aqua deployments within a specified compartment and/or project.
    get_deployment_config(self, model_id: str) -> Dict:
        Gets the deployment config of given Aqua model.

    Note:
        Use `ads aqua deployment <method_name> --help` to get more details on the parameters available.
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    @telemetry(entry_point="plugin=deployment&action=create", name="aqua")
    def create(
        self,
        model_id: str,
        instance_shape: str,
        display_name: str,
        instance_count: int = None,
        log_group_id: str = None,
        access_log_id: str = None,
        predict_log_id: str = None,
        compartment_id: str = None,
        project_id: str = None,
        description: str = None,
        bandwidth_mbps: int = None,
        web_concurrency: int = None,
        server_port: int = None,
        health_check_port: int = None,
        env_var: Dict = None,
        container_family: str = None,
        memory_in_gbs: Optional[float] = None,
        ocpus: Optional[float] = None,
        model_file: Optional[str] = None,
        private_endpoint_id: Optional[str] = None,
        container_image_uri: Optional[None] = None,
        cmd_var: List[str] = None,
    ) -> "AquaDeployment":
        """
        Creates a new Aqua deployment

        Parameters
        ----------
        model_id: str
            The model OCID to deploy.
        compartment_id: str
            The compartment OCID
        project_id: str
            Target project to list deployments from.
        display_name: str
            The name of model deployment.
        description: str
            The description of the deployment.
        instance_count: (int, optional). Defaults to 1.
            The number of instance used for deployment.
        instance_shape: (str).
            The shape of the instance used for deployment.
        log_group_id: (str)
            The oci logging group id. The access log and predict log share the same log group.
        access_log_id: (str).
            The access log OCID for the access logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        predict_log_id: (str).
            The predict log OCID for the predict logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        bandwidth_mbps: (int). Defaults to 10.
            The bandwidth limit on the load balancer in Mbps.
        web_concurrency: str
            The number of worker processes/threads to handle incoming requests
        with_bucket_uri(bucket_uri)
            Sets the bucket uri when uploading large size model.
        server_port: (int).
            The server port for docker container image.
        health_check_port: (int).
            The health check port for docker container image.
        env_var : dict, optional
            Environment variable for the deployment, by default None.
        container_family: str
            The image family of model deployment container runtime.
        memory_in_gbs: float
            The memory in gbs for the shape selected.
        ocpus: float
            The ocpu count for the shape selected.
        model_file: str
            The file used for model deployment.
        private_endpoint_id: str
            The private endpoint id of model deployment.
        container_image_uri: str
            The image of model deployment container runtime, ignored for service managed containers.
            Required parameter for BYOC based deployments if this parameter was not set during model registration.
        cmd_var: List[str]
            The cmd of model deployment container runtime.
        Returns
        -------
        AquaDeployment
            An Aqua deployment instance

        """
        # TODO validate if the service model has no artifact and if it requires import step before deployment.
        # Create a model catalog entry in the user compartment
        aqua_model = AquaModelApp().create(
            model_id=model_id, compartment_id=compartment_id, project_id=project_id
        )

        tags = {}
        for tag in [
            Tags.AQUA_SERVICE_MODEL_TAG,
            Tags.AQUA_FINE_TUNED_MODEL_TAG,
            Tags.AQUA_TAG,
        ]:
            if tag in aqua_model.freeform_tags:
                tags[tag] = aqua_model.freeform_tags[tag]

        tags.update({Tags.AQUA_MODEL_NAME_TAG: aqua_model.display_name})
        tags.update({Tags.TASK: aqua_model.freeform_tags.get(Tags.TASK, None)})

        # Set up info to get deployment config
        config_source_id = model_id
        model_name = aqua_model.display_name

        is_fine_tuned_model = Tags.AQUA_FINE_TUNED_MODEL_TAG in aqua_model.freeform_tags

        if is_fine_tuned_model:
            try:
                config_source_id = aqua_model.custom_metadata_list.get(
                    FineTuneCustomMetadata.FINE_TUNE_SOURCE
                ).value
                model_name = aqua_model.custom_metadata_list.get(
                    FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME
                ).value
            except ValueError as err:
                raise AquaValueError(
                    f"Either {FineTuneCustomMetadata.FINE_TUNE_SOURCE} or {FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME} is missing "
                    f"from custom metadata for the model {config_source_id}"
                ) from err

        # set up env and cmd var
        if not env_var:
            env_var = {}
        if not cmd_var:
            cmd_var = []

        try:
            model_path_prefix = aqua_model.custom_metadata_list.get(
                MODEL_BY_REFERENCE_OSS_PATH_KEY
            ).value.rstrip("/")
        except ValueError as err:
            raise AquaValueError(
                f"{MODEL_BY_REFERENCE_OSS_PATH_KEY} key is not available in the custom metadata field."
            ) from err

        if ObjectStorageDetails.is_oci_path(model_path_prefix):
            os_path = ObjectStorageDetails.from_path(model_path_prefix)
            model_path_prefix = os_path.filepath.rstrip("/")

        env_var.update({"BASE_MODEL": f"{model_path_prefix}"})

        if is_fine_tuned_model:
            _, fine_tune_output_path = get_model_by_reference_paths(
                aqua_model.model_file_description
            )

            if not fine_tune_output_path:
                raise AquaValueError(
                    "Fine tuned output path is not available in the model artifact."
                )

            os_path = ObjectStorageDetails.from_path(fine_tune_output_path)
            fine_tune_output_path = os_path.filepath.rstrip("/")

            env_var.update({"FT_MODEL": f"{fine_tune_output_path}"})

        container_type_key = self._get_container_type_key(
            model=aqua_model, container_family=container_family
        )

        container_image_uri = container_image_uri or get_container_image(
            container_type=container_type_key
        )
        if not container_image_uri:
            try:
                container_image_uri = aqua_model.custom_metadata_list.get(
                    AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME
                ).value
            except ValueError as err:
                raise AquaValueError(
                    f"{AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME} key is not available in the custom metadata "
                    f"field. Either re-register the model with custom container URI, or set container_image_uri "
                    f"parameter when creating this deployment."
                ) from err
        logging.info(
            f"Aqua Image used for deploying {aqua_model.id} : {container_image_uri}"
        )

        try:
            cmd_var_string = aqua_model.custom_metadata_list.get(
                AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME
            ).value
            default_cmd_var = shlex.split(cmd_var_string)
            if default_cmd_var:
                cmd_var = validate_cmd_var(default_cmd_var, cmd_var)
            logging.info(f"CMD used for deploying {aqua_model.id} :{cmd_var}")
        except ValueError:
            logging.debug(
                f"CMD will be ignored for this deployment as {AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME} "
                f"key is not available in the custom metadata field for this model."
            )
        except Exception as e:
            logging.error(
                f"There was an issue processing CMD arguments. Error: {str(e)}"
            )

        model_formats_str = aqua_model.freeform_tags.get(
            Tags.MODEL_FORMAT, ModelFormat.SAFETENSORS.value
        ).upper()
        model_format = model_formats_str.split(",")

        # Figure out a better way to handle this in future release
        if (
            ModelFormat.GGUF.value in model_format
            and container_type_key.lower()
            == InferenceContainerTypeFamily.AQUA_LLAMA_CPP_CONTAINER_FAMILY
        ):
            if model_file is not None:
                logger.info(
                    f"Overriding {model_file} as model_file for model {aqua_model.id}."
                )
            else:
                try:
                    model_file = aqua_model.custom_metadata_list.get(
                        AQUA_MODEL_ARTIFACT_FILE
                    ).value
                except ValueError as err:
                    raise AquaValueError(
                        f"{AQUA_MODEL_ARTIFACT_FILE} key is not available in the custom metadata field "
                        f"for model {aqua_model.id}. Either register the model with a default model_file or pass "
                        f"as a parameter when creating a deployment."
                    ) from err

            env_var.update({"BASE_MODEL_FILE": f"{model_file}"})
            tags.update({Tags.MODEL_ARTIFACT_FILE: model_file})

        # todo: use AquaContainerConfig.from_container_index_json instead.
        # Fetch the startup cli command for the container
        # container_index.json will have "containerSpec" section which will provide the cli params for
        # a given container family
        container_config = get_container_config()
        container_spec = container_config.get(ContainerSpec.CONTAINER_SPEC, {}).get(
            container_type_key, {}
        )
        # these params cannot be overridden for Aqua deployments
        params = container_spec.get(ContainerSpec.CLI_PARM, "")
        server_port = server_port or container_spec.get(
            ContainerSpec.SERVER_PORT
        )  # Give precendece to the input parameter
        health_check_port = health_check_port or container_spec.get(
            ContainerSpec.HEALTH_CHECK_PORT
        )  # Give precendece to the input parameter

        deployment_config = self.get_deployment_config(config_source_id)

        config_params = (
            deployment_config.get("configuration", UNKNOWN_DICT)
            .get(instance_shape, UNKNOWN_DICT)
            .get("parameters", UNKNOWN_DICT)
            .get(get_container_params_type(container_type_key), UNKNOWN)
        )

        # validate user provided params
        user_params = env_var.get("PARAMS", UNKNOWN)
        if user_params:
            # todo: remove this check in the future version, logic to be moved to container_index
            if (
                container_type_key.lower()
                == InferenceContainerTypeFamily.AQUA_LLAMA_CPP_CONTAINER_FAMILY
            ):
                # AQUA_LLAMA_CPP_CONTAINER_FAMILY container uses uvicorn that required model/server params
                # to be set as env vars
                raise AquaValueError(
                    f"Currently, parameters cannot be overridden for the container: {container_image_uri}. Please proceed "
                    f"with deployment without parameter overrides."
                )

            restricted_params = self._find_restricted_params(
                params, user_params, container_type_key
            )
            if restricted_params:
                raise AquaValueError(
                    f"Parameters {restricted_params} are set by Aqua "
                    f"and cannot be overridden or are invalid."
                )

        deployment_params = get_combined_params(config_params, user_params)

        params = f"{params} {deployment_params}".strip()
        if params:
            env_var.update({"PARAMS": params})

        for env in container_spec.get(ContainerSpec.ENV_VARS, []):
            if isinstance(env, dict):
                for key, _items in env.items():
                    if key not in env_var:
                        env_var.update(env)

        logging.info(f"Env vars used for deploying {aqua_model.id} :{env_var}")

        # Start model deployment
        # configure model deployment infrastructure
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_project_id(project_id)
            .with_compartment_id(compartment_id)
            .with_shape_name(instance_shape)
            .with_bandwidth_mbps(bandwidth_mbps)
            .with_replica(instance_count)
            .with_web_concurrency(web_concurrency)
            .with_private_endpoint_id(private_endpoint_id)
            .with_access_log(
                log_group_id=log_group_id,
                log_id=access_log_id,
            )
            .with_predict_log(
                log_group_id=log_group_id,
                log_id=predict_log_id,
            )
        )
        if memory_in_gbs and ocpus and infrastructure.shape_name.endswith("Flex"):
            infrastructure.with_shape_config_details(
                ocpus=ocpus,
                memory_in_gbs=memory_in_gbs,
            )
        # configure model deployment runtime
        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(container_image_uri)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_deployment_mode(ModelDeploymentMode.HTTPS)
            .with_model_uri(aqua_model.id)
            .with_region(self.region)
            .with_overwrite_existing_artifact(True)
            .with_remove_existing_artifact(True)
        )
        if cmd_var:
            container_runtime.with_cmd(cmd_var)

        # configure model deployment and deploy model on container runtime
        deployment = (
            ModelDeployment()
            .with_display_name(display_name)
            .with_description(description)
            .with_freeform_tags(**tags)
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        ).deploy(wait_for_completion=False)

        model_type = (
            AQUA_MODEL_TYPE_CUSTOM if is_fine_tuned_model else AQUA_MODEL_TYPE_SERVICE
        )
        deployment_id = deployment.dsc_model_deployment.id
        # we arbitrarily choose last 8 characters of OCID to identify MD in telemetry
        telemetry_kwargs = {"ocid": get_ocid_substring(deployment_id, key_len=8)}

        # tracks unique deployments that were created in the user compartment
        self.telemetry.record_event_async(
            category=f"aqua/{model_type}/deployment",
            action="create",
            detail=model_name,
            **telemetry_kwargs,
        )
        # tracks the shape used for deploying the custom or service models by name
        self.telemetry.record_event_async(
            category=f"aqua/{model_type}/deployment/create",
            action="shape",
            detail=instance_shape,
            value=model_name,
        )

        return AquaDeployment.from_oci_model_deployment(
            deployment.dsc_model_deployment, self.region
        )

    @staticmethod
    def _get_container_type_key(model: DataScienceModel, container_family: str) -> str:
        container_type_key = UNKNOWN
        if container_family:
            container_type_key = container_family
        else:
            try:
                container_type_key = model.custom_metadata_list.get(
                    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME
                ).value
            except ValueError as err:
                raise AquaValueError(
                    f"{AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME} key is not available in the custom metadata field "
                    f"for model {model.id}. For unverified Aqua models, {AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME} should be"
                    f"set and value can be one of {', '.join(InferenceContainerTypeFamily.values())}."
                ) from err

        return container_type_key

    @telemetry(entry_point="plugin=deployment&action=list", name="aqua")
    def list(self, **kwargs) -> List["AquaDeployment"]:
        """List Aqua model deployments in a given compartment and under certain project.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id and project_id,
            for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

        Returns
        -------
        List[AquaDeployment]:
            The list of the Aqua model deployments.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        model_deployments = self.list_resource(
            self.ds_client.list_model_deployments,
            compartment_id=compartment_id,
            **kwargs,
        )

        results = []
        for model_deployment in model_deployments:
            oci_aqua = (
                (
                    Tags.AQUA_TAG in model_deployment.freeform_tags
                    or Tags.AQUA_TAG.lower() in model_deployment.freeform_tags
                )
                if model_deployment.freeform_tags
                else False
            )

            if oci_aqua:
                results.append(
                    AquaDeployment.from_oci_model_deployment(
                        model_deployment, self.region
                    )
                )

                # log telemetry if MD is in active or failed state
                deployment_id = model_deployment.id
                state = model_deployment.lifecycle_state.upper()
                if state in ["ACTIVE", "FAILED"]:
                    # tracks unique deployments that were listed in the user compartment
                    # we arbitrarily choose last 8 characters of OCID to identify MD in telemetry
                    self.telemetry.record_event_async(
                        category="aqua/deployment",
                        action="list",
                        detail=get_ocid_substring(deployment_id, key_len=8),
                        value=state,
                    )

        # tracks number of times deployment listing was called
        self.telemetry.record_event_async(category="aqua/deployment", action="list")

        return results

    @telemetry(entry_point="plugin=deployment&action=delete", name="aqua")
    def delete(self,model_deployment_id:str):
        return self.ds_client.delete_model_deployment(model_deployment_id=model_deployment_id).data

    @telemetry(entry_point="plugin=deployment&action=deactivate",name="aqua")
    def deactivate(self,model_deployment_id:str):
        return self.ds_client.deactivate_model_deployment(model_deployment_id=model_deployment_id).data

    @telemetry(entry_point="plugin=deployment&action=activate",name="aqua")
    def activate(self,model_deployment_id:str):
        return self.ds_client.activate_model_deployment(model_deployment_id=model_deployment_id).data

    @telemetry(entry_point="plugin=deployment&action=get", name="aqua")
    def get(self, model_deployment_id: str, **kwargs) -> "AquaDeploymentDetail":
        """Gets the information of Aqua model deployment.

        Parameters
        ----------
        model_deployment_id: str
            The OCID of the Aqua model deployment.
        kwargs
            Keyword arguments, for `get_model_deployment
            <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.get_model_deployment>`_

        Returns
        -------
        AquaDeploymentDetail:
            The instance of the Aqua model deployment details.
        """
        model_deployment = self.ds_client.get_model_deployment(
            model_deployment_id=model_deployment_id, **kwargs
        ).data

        oci_aqua = (
            (
                Tags.AQUA_TAG in model_deployment.freeform_tags
                or Tags.AQUA_TAG.lower() in model_deployment.freeform_tags
            )
            if model_deployment.freeform_tags
            else False
        )

        if not oci_aqua:
            raise AquaRuntimeError(
                f"Target deployment {model_deployment_id} is not Aqua deployment."
            )

        log_id = ""
        log_group_id = ""
        log_name = ""
        log_group_name = ""

        logs = (
            model_deployment.category_log_details.access
            or model_deployment.category_log_details.predict
        )
        if logs:
            log_id = logs.log_id
            log_group_id = logs.log_group_id
        if log_id:
            log_name = get_resource_name(log_id)
        if log_group_id:
            log_group_name = get_resource_name(log_group_id)

        log_group_url = get_log_links(region=self.region, log_group_id=log_group_id)
        log_url = get_log_links(
            region=self.region,
            log_group_id=log_group_id,
            log_id=log_id,
            compartment_id=model_deployment.compartment_id,
            source_id=model_deployment.id,
        )

        return AquaDeploymentDetail(
            **vars(
                AquaDeployment.from_oci_model_deployment(model_deployment, self.region)
            ),
            log_group=AquaResourceIdentifier(
                log_group_id, log_group_name, log_group_url
            ),
            log=AquaResourceIdentifier(log_id, log_name, log_url),
        )

    @telemetry(
        entry_point="plugin=deployment&action=get_deployment_config", name="aqua"
    )
    def get_deployment_config(self, model_id: str) -> Dict:
        """Gets the deployment config of given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            A dict of allowed deployment configs.
        """
        config = self.get_config(model_id, AQUA_MODEL_DEPLOYMENT_CONFIG)
        if not config:
            logger.debug(
                f"Deployment config for custom model: {model_id} is not available."
            )
        return config

    def get_deployment_default_params(
        self,
        model_id: str,
        instance_shape: str,
    ) -> List[str]:
        """Gets the default params set in the deployment configs for the given model and instance shape.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        instance_shape: (str).
            The shape of the instance used for deployment.

        Returns
        -------
        List[str]:
            List of parameters from the loaded from deployment config json file. If not available, then an empty list
            is returned.

        """
        default_params = []
        model = DataScienceModel.from_id(model_id)
        try:
            container_type_key = model.custom_metadata_list.get(
                AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME
            ).value
        except ValueError:
            container_type_key = UNKNOWN
            logger.debug(
                f"{AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME} key is not available in the custom metadata field for model {model_id}."
            )

        if (
            container_type_key
            and container_type_key in InferenceContainerTypeFamily.values()
        ):
            deployment_config = self.get_deployment_config(model_id)
            config_params = (
                deployment_config.get("configuration", UNKNOWN_DICT)
                .get(instance_shape, UNKNOWN_DICT)
                .get("parameters", UNKNOWN_DICT)
                .get(get_container_params_type(container_type_key), UNKNOWN)
            )
            if config_params:
                params_list = get_params_list(config_params)
                restricted_params_set = get_restricted_params_by_container(
                    container_type_key
                )

                # remove restricted params from the list as user cannot override them during deployment
                for params in params_list:
                    if params.split()[0] not in restricted_params_set:
                        default_params.append(params)

        return default_params

    def validate_deployment_params(
        self,
        model_id: str,
        params: List[str] = None,
        container_family: str = None,
    ) -> Dict:
        """Validate if the deployment parameters passed by the user can be overridden. Parameter values are not
        validated, only param keys are validated.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.
        params : List[str], optional
            Params passed by the user.
        container_family: str
            The image family of model deployment container runtime. Required for unverified Aqua models.

        Returns
        -------
            Return a list of restricted params.

        """
        restricted_params = []
        if params:
            model = DataScienceModel.from_id(model_id)
            container_type_key = self._get_container_type_key(
                model=model, container_family=container_family
            )

            container_config = get_container_config()
            container_spec = container_config.get(ContainerSpec.CONTAINER_SPEC, {}).get(
                container_type_key, {}
            )
            cli_params = container_spec.get(ContainerSpec.CLI_PARM, "")

            restricted_params = self._find_restricted_params(
                cli_params, params, container_type_key
            )

        if restricted_params:
            raise AquaValueError(
                f"Parameters {restricted_params} are set by Aqua "
                f"and cannot be overridden or are invalid."
            )
        return {"valid": True}

    @staticmethod
    def _find_restricted_params(
        default_params: Union[str, List[str]],
        user_params: Union[str, List[str]],
        container_family: str,
    ) -> List[str]:
        """Returns a list of restricted params that user chooses to override when creating an Aqua deployment.
        The default parameters coming from the container index json file cannot be overridden.

        Parameters
        ----------
        default_params:
            Inference container parameter string with default values.
        user_params:
            Inference container parameter string with user provided values.
        container_family: str
            The image family of model deployment container runtime.

        Returns
        -------
            A list with params keys common between params1 and params2.

        """
        restricted_params = []
        if default_params and user_params:
            default_params_dict = get_params_dict(default_params)
            user_params_dict = get_params_dict(user_params)

            restricted_params_set = get_restricted_params_by_container(container_family)
            for key, _items in user_params_dict.items():
                if key in default_params_dict or key in restricted_params_set:
                    restricted_params.append(key.lstrip("-"))

        return restricted_params
