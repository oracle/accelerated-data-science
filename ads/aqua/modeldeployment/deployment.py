#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import shlex
from typing import Dict, List, Optional, Union

from pydantic import ValidationError

from ads.aqua.app import AquaApp, logger
from ads.aqua.common.entities import ContainerSpec
from ads.aqua.common.enums import InferenceContainerTypeFamily, Tags
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.aqua.common.utils import (
    build_pydantic_error_message,
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
    AQUA_MODEL_TYPE_MULTI,
    AQUA_MODEL_TYPE_SERVICE,
    AQUA_MULTI_MODEL_CONFIG,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    UNKNOWN,
    UNKNOWN_DICT,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.finetuning.finetuning import FineTuneCustomMetadata
from ads.aqua.model import AquaModelApp
from ads.aqua.model.constants import ModelCustomMetadataFields
from ads.aqua.modeldeployment.entities import (
    AquaDeployment,
    AquaDeploymentConfig,
    AquaDeploymentDetail,
    ConfigurationItem,
    CreateModelDeploymentDetails,
    ModelDeploymentConfigSummary,
)
from ads.aqua.modeldeployment.utils import MultiModelDeploymentConfigLoader
from ads.aqua.ui import ModelFormat
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import get_log_links
from ads.config import (
    AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME,
    AQUA_MODEL_DEPLOYMENT_CONFIG,
    COMPARTMENT_OCID,
    PROJECT_OCID,
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
    get_deployment_config(self, model_id: str) -> AquaDeploymentConfig:
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
        create_deployment_details: Optional[CreateModelDeploymentDetails] = None,
        **kwargs,
    ) -> "AquaDeployment":
        """
        Creates a new Aqua model deployment.

        Parameters
        ----------
        create_deployment_details : CreateModelDeploymentDetails, optional
            An instance of CreateModelDeploymentDetails containing all required and optional
            fields for creating a model deployment via Aqua.
        **kwargs:
            Keyword arguments used to construct a CreateModelDeploymentDetails instance if one
            is not provided.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        # Build deployment details from kwargs if not explicitly provided.
        if create_deployment_details is None:
            try:
                create_deployment_details = CreateModelDeploymentDetails(**kwargs)
            except ValidationError as ex:
                custom_errors = build_pydantic_error_message(ex)
                raise AquaValueError(
                    f"Invalid parameters for creating a model deployment. Error details: {custom_errors}."
                ) from ex

        # Extract model_id from the provided deployment details.
        model_id = create_deployment_details.model_id

        # If a single model is provided, delegate to `create` method
        if (
            not model_id
            and create_deployment_details.models
            and len(create_deployment_details.models) == 1
        ):
            single_model = create_deployment_details.models[0]
            logger.info(
                f"Single model ({single_model.model_id}) provided. "
                "Delegating to single model creation method."
            )
            model_id = single_model.model_id

        # Set defaults for compartment and project if not provided.
        compartment_id = create_deployment_details.compartment_id or COMPARTMENT_OCID
        project_id = create_deployment_details.project_id or PROJECT_OCID
        freeform_tags = create_deployment_details.freeform_tags
        defined_tags = create_deployment_details.defined_tags

        # Create an AquaModelApp instance once to perform the deployment creation.
        model_app = AquaModelApp()
        if model_id:
            aqua_model = model_app.create(
                model_id=model_id,
                compartment_id=compartment_id,
                project_id=project_id,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
            )
        else:
            aqua_model = model_app.create_multi(
                models=create_deployment_details.models,
                compartment_id=compartment_id,
                project_id=project_id,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
            )
            return self._create_multi(
                aqua_model=aqua_model,
                create_deployment_details=create_deployment_details,
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
        tags.update({Tags.TASK: aqua_model.freeform_tags.get(Tags.TASK, UNKNOWN)})

        # Set up info to get deployment config
        config_source_id = create_deployment_details.model_id
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
        env_var = create_deployment_details.env_var or {}
        cmd_var = create_deployment_details.cmd_var or []

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
            model=aqua_model,
            container_family=create_deployment_details.container_family,
        )

        container_image_uri = (
            create_deployment_details.container_image_uri
            or get_container_image(container_type=container_type_key)
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
        logger.info(
            f"Aqua Image used for deploying {aqua_model.id} : {container_image_uri}"
        )

        try:
            cmd_var_string = aqua_model.custom_metadata_list.get(
                AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME
            ).value
            default_cmd_var = shlex.split(cmd_var_string)
            if default_cmd_var:
                cmd_var = validate_cmd_var(default_cmd_var, cmd_var)
            logger.info(f"CMD used for deploying {aqua_model.id} :{cmd_var}")
        except ValueError:
            logger.debug(
                f"CMD will be ignored for this deployment as {AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME} "
                f"key is not available in the custom metadata field for this model."
            )
        except Exception as e:
            logger.error(
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
            model_file = create_deployment_details.model_file
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
        server_port = create_deployment_details.server_port or container_spec.get(
            ContainerSpec.SERVER_PORT
        )  # Give precedence to the input parameter
        health_check_port = (
            create_deployment_details.health_check_port
            or container_spec.get(ContainerSpec.HEALTH_CHECK_PORT)
        )  # Give precedence to the input parameter

        deployment_config = self.get_deployment_config(config_source_id)

        config_params = deployment_config.configuration.get(
            create_deployment_details.instance_shape, ConfigurationItem()
        ).parameters.get(get_container_params_type(container_type_key), UNKNOWN)

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
                for key, _ in env.items():
                    if key not in env_var:
                        env_var.update(env)

        logger.info(f"Env vars used for deploying {aqua_model.id} :{env_var}")

        tags = {**tags, **(create_deployment_details.freeform_tags or {})}
        model_type = (
            AQUA_MODEL_TYPE_CUSTOM if is_fine_tuned_model else AQUA_MODEL_TYPE_SERVICE
        )

        return self._create_deployment(
            create_deployment_details=create_deployment_details,
            aqua_model_id=aqua_model.id,
            model_name=model_name,
            model_type=model_type,
            container_image_uri=container_image_uri,
            server_port=server_port,
            health_check_port=health_check_port,
            env_var=env_var,
            tags=tags,
            cmd_var=cmd_var,
        )

    def _create_multi(
        self,
        aqua_model: DataScienceModel,
        create_deployment_details: CreateModelDeploymentDetails,
    ) -> Dict:
        """Builds the environment variables required by multi deployment container and creates the deployment.

        Parameters
        ----------
        aqua_model : DataScienceModel
            An instance of Aqua data science model.
        create_deployment_details : CreateModelDeploymentDetails
            An instance of CreateModelDeploymentDetails containing all required and optional
            fields for creating a model deployment via Aqua.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        model_config = []
        model_name_list = []
        env_var = {**(create_deployment_details.env_var or UNKNOWN_DICT)}

        container_type_key = self._get_container_type_key(
            model=aqua_model,
            container_family=create_deployment_details.container_family,
        )
        container_config = get_container_config()
        container_spec = container_config.get(
            ContainerSpec.CONTAINER_SPEC, UNKNOWN_DICT
        ).get(container_type_key, UNKNOWN_DICT)

        container_params = container_spec.get(ContainerSpec.CLI_PARM, UNKNOWN).strip()

        for idx, model in enumerate(create_deployment_details.models):
            user_params = (
                "".join(f"{name} {value}" for name, value in model.env_var.items())
                if model.env_var
                else UNKNOWN
            )
            if user_params:
                restricted_params = self._find_restricted_params(
                    container_params, user_params, container_type_key
                )
                if restricted_params:
                    raise AquaValueError(
                        f"Parameters {restricted_params} are set by Aqua "
                        f"and cannot be overridden or are invalid."
                        f"Select other parameters for model {model.model_id}."
                    )

            params = container_params
            deployment_config = self.get_deployment_config(model.model_id)
            multi_model_deployment = deployment_config.configuration.get(
                create_deployment_details.instance_shape, ConfigurationItem()
            ).multi_model_deployment
            for item in multi_model_deployment:
                if (
                    model.gpu_count
                    and item.gpu_count
                    and item.gpu_count == model.gpu_count
                ):
                    config_parameters = item.parameters.get(
                        get_container_params_type(container_type_key), UNKNOWN
                    )
                    params = f"{params} {get_combined_params(config_parameters, user_params)}".strip()
                    break

            artifact_location_key = (
                f"{ModelCustomMetadataFields.ARTIFACT_LOCATION}-{idx}"
            )
            artifact_path_prefix = aqua_model.custom_metadata_list.get(
                artifact_location_key
            ).value.rstrip("/")
            if ObjectStorageDetails.is_oci_path(artifact_path_prefix):
                os_path = ObjectStorageDetails.from_path(artifact_path_prefix)
                artifact_path_prefix = os_path.filepath.rstrip("/")

            model_config.append({"params": params, "model_path": artifact_path_prefix})

            model_name_key = f"model-name-{idx}"
            model_name_list.append(
                aqua_model.custom_metadata_list.get(model_name_key).value
            )

        env_var.update({AQUA_MULTI_MODEL_CONFIG: json.dumps({"models": model_config})})
        logger.info(f"Env vars used for deploying {aqua_model.id} : {env_var}.")

        container_image_uri = (
            create_deployment_details.container_image_uri
            or get_container_image(container_type=container_type_key)
        )
        server_port = create_deployment_details.server_port or container_spec.get(
            ContainerSpec.SERVER_PORT
        )
        health_check_port = (
            create_deployment_details.health_check_port
            or container_spec.get(ContainerSpec.HEALTH_CHECK_PORT)
        )
        tags = {
            Tags.AQUA_MODEL_ID_TAG: aqua_model.id,
            Tags.MULTIMODEL_TYPE_TAG: "true",
            Tags.AQUA_TAG: "active",
            **(create_deployment_details.freeform_tags or UNKNOWN_DICT),
        }

        model_name = ", ".join(model_name_list)

        aqua_deployment = self._create_deployment(
            create_deployment_details=create_deployment_details,
            aqua_model_id=aqua_model.id,
            model_name=model_name,
            model_type=AQUA_MODEL_TYPE_MULTI,
            container_image_uri=container_image_uri,
            server_port=server_port,
            health_check_port=health_check_port,
            env_var=env_var,
            tags=tags,
        )
        aqua_deployment.models = create_deployment_details.models
        return aqua_deployment

    def _create_deployment(
        self,
        create_deployment_details: CreateModelDeploymentDetails,
        aqua_model_id: str,
        model_name: str,
        model_type: str,
        container_image_uri: str,
        server_port: str,
        health_check_port: str,
        env_var: dict,
        tags: dict,
        cmd_var: Optional[dict] = None,
    ):
        """Creates data science model deployment.

        Parameters
        ----------
        create_deployment_details : CreateModelDeploymentDetails
            An instance of CreateModelDeploymentDetails containing all required and optional
            fields for creating a model deployment via Aqua.
        aqua_model_id: str
            The id of the aqua model to be deployed.
        model_name: str
            The name of the aqua model to be deployed. If it's multi model deployment, it is a list of model names.
        model_type: str
            The type of aqua model to be deployed. Allowed values are: `custom`, `service` and `multi_model`.
        container_image_uri: str
            The container image uri to deploy the model.
        server_port: str
            The service port of the container image.
        health_check_port: str
            The health check port of the container image.
        env_var: dict
            The environment variables input for the deployment.
        tags: dict
            The tags input for the deployment.
        cmd_var: dict, optional
            The cmd arguments input for the deployment.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        # Start model deployment
        # configure model deployment infrastructure
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_project_id(create_deployment_details.project_id or PROJECT_OCID)
            .with_compartment_id(
                create_deployment_details.compartment_id or COMPARTMENT_OCID
            )
            .with_shape_name(create_deployment_details.instance_shape)
            .with_bandwidth_mbps(create_deployment_details.bandwidth_mbps)
            .with_replica(create_deployment_details.instance_count)
            .with_web_concurrency(create_deployment_details.web_concurrency)
            .with_private_endpoint_id(create_deployment_details.private_endpoint_id)
            .with_access_log(
                log_group_id=create_deployment_details.log_group_id,
                log_id=create_deployment_details.access_log_id,
            )
            .with_predict_log(
                log_group_id=create_deployment_details.log_group_id,
                log_id=create_deployment_details.predict_log_id,
            )
        )
        if (
            create_deployment_details.memory_in_gbs
            and create_deployment_details.ocpus
            and infrastructure.shape_name.endswith("Flex")
        ):
            infrastructure.with_shape_config_details(
                ocpus=create_deployment_details.ocpus,
                memory_in_gbs=create_deployment_details.memory_in_gbs,
            )
        # configure model deployment runtime
        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(container_image_uri)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_deployment_mode(ModelDeploymentMode.HTTPS)
            .with_model_uri(aqua_model_id)
            .with_region(self.region)
            .with_overwrite_existing_artifact(True)
            .with_remove_existing_artifact(True)
        )
        if cmd_var:
            container_runtime.with_cmd(cmd_var)

        # configure model deployment and deploy model on container runtime
        deployment = (
            ModelDeployment()
            .with_display_name(create_deployment_details.display_name)
            .with_description(create_deployment_details.description)
            .with_freeform_tags(**tags)
            .with_defined_tags(**(create_deployment_details.defined_tags or {}))
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        ).deploy(wait_for_completion=False)

        deployment_id = deployment.id
        logger.info(
            f"Aqua model deployment {deployment_id} created for model {aqua_model_id}."
        )

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
            detail=create_deployment_details.instance_shape,
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

        logger.info(
            f"Fetched {len(results)} model deployments from compartment_id={compartment_id}."
        )
        # tracks number of times deployment listing was called
        self.telemetry.record_event_async(category="aqua/deployment", action="list")

        return results

    @telemetry(entry_point="plugin=deployment&action=delete", name="aqua")
    def delete(self, model_deployment_id: str):
        logger.info(f"Deleting model deployment {model_deployment_id}.")
        return self.ds_client.delete_model_deployment(
            model_deployment_id=model_deployment_id
        ).data

    @telemetry(entry_point="plugin=deployment&action=deactivate", name="aqua")
    def deactivate(self, model_deployment_id: str):
        logger.info(f"Deactivating model deployment {model_deployment_id}.")
        return self.ds_client.deactivate_model_deployment(
            model_deployment_id=model_deployment_id
        ).data

    @telemetry(entry_point="plugin=deployment&action=activate", name="aqua")
    def activate(self, model_deployment_id: str):
        logger.info(f"Activating model deployment {model_deployment_id}.")
        return self.ds_client.activate_model_deployment(
            model_deployment_id=model_deployment_id
        ).data

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
        logger.info(f"Fetching model deployment details for {model_deployment_id}.")

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
                f"Target deployment {model_deployment_id} is not Aqua deployment as it does not contain "
                f"{Tags.AQUA_TAG} tag."
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
    def get_deployment_config(self, model_id: str) -> AquaDeploymentConfig:
        """Gets the deployment config of given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        AquaDeploymentConfig:
            An instance of AquaDeploymentConfig.
        """
        config = self.get_config(model_id, AQUA_MODEL_DEPLOYMENT_CONFIG)
        if not config:
            logger.debug(
                f"Deployment config for custom model: {model_id} is not available. Use defaults."
            )
        return AquaDeploymentConfig(**(config or UNKNOWN_DICT))

    @telemetry(
        entry_point="plugin=deployment&action=get_multimodel_deployment_config",
        name="aqua",
    )
    def get_multimodel_deployment_config(
        self, model_ids: List[str], primary_model_id: Optional[str] = None
    ) -> ModelDeploymentConfigSummary:
        """
        Retrieves the deployment configuration for multiple Aqua models and calculates
        the GPU allocations for all compatible shapes.

        If no primary Aqua model id provided, gpu count for each compatible shape will be evenly allocated.
        If provided, gpu count for each compatible shape will be prioritized for primary model.

        For example, there is one compatible shape "BM.GPU.H100.8" for three models A, B, C, and each model has a gpu count as below:

        A - BM.GPU.H100.8 - 1, 2, 4, 8
        B - BM.GPU.H100.8 - 1, 2, 4, 8
        C - BM.GPU.H100.8 - 1, 2, 4, 8

        If no primary model is provided, the gpu allocation for A, B, C could be [2, 4, 2], [2, 2, 4] or [4, 2, 2]
        If B is the primary model, the gpu allocation is [2, 4, 2] as B always gets the maximum gpu count.

        Parameters
        ----------
        model_ids : List[str]
            A list of OCIDs for the Aqua models.
        primary_model_id : Optional[str]
            The OCID of the primary Aqua model. If provided, GPU allocation will prioritize
            this model. Otherwise, GPUs will be evenly allocated.

        Returns
        -------
        ModelDeploymentConfigSummary
            A summary of the model deployment configurations and GPU allocations.
        """

        return MultiModelDeploymentConfigLoader(self).load(model_ids, primary_model_id)

    def get_deployment_default_params(
        self,
        model_id: str,
        instance_shape: str,
        gpu_count: int = None,
    ) -> List[str]:
        """Gets the default params set in the deployment configs for the given model and instance shape.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        instance_shape: (str).
            The shape of the instance used for deployment.

        gpu_count: (int, optional).
            The number of GPUs used by the Aqua model. Defaults to None.

        Returns
        -------
        List[str]:
            List of parameters from the loaded from deployment config json file. If not available, then an empty list
            is returned.

        """
        default_params = []
        config_params = {}
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

            instance_shape_config = deployment_config.configuration.get(
                instance_shape, ConfigurationItem()
            )

            if instance_shape_config.multi_model_deployment and gpu_count:
                gpu_params = instance_shape_config.multi_model_deployment

                for gpu_config in gpu_params:
                    if gpu_config.gpu_count == gpu_count:
                        config_params = gpu_config.parameters.get(
                            get_container_params_type(container_type_key), UNKNOWN
                        )
                        break

            else:
                config_params = instance_shape_config.parameters.get(
                    get_container_params_type(container_type_key), UNKNOWN
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
