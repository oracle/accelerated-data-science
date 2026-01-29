#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import re
import shlex
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from cachetools import TTLCache, cached
from oci.data_science.models import ModelDeploymentShapeSummary
from pydantic import ValidationError
from rich.table import Table

from ads.aqua.app import AquaApp, logger
from ads.aqua.common.entities import (
    AquaMultiModelRef,
    ComputeShapeSummary,
    ContainerPath,
)
from ads.aqua.common.enums import InferenceContainerTypeFamily, ModelFormat, Tags
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.aqua.common.utils import (
    DEFINED_METADATA_TO_FILE_MAP,
    build_params_string,
    build_pydantic_error_message,
    find_restricted_params,
    get_container_env_type,
    get_container_params_type,
    get_ocid_substring,
    get_params_dict,
    get_params_list,
    get_preferred_compatible_family,
    get_resource_name,
    get_restricted_params_by_container,
    is_valid_ocid,
    load_gpu_shapes_index,
    validate_cmd_var,
)
from ads.aqua.config.container_config import AquaContainerConfig, Usage
from ads.aqua.constants import (
    AQUA_MODEL_ARTIFACT_FILE,
    AQUA_MODEL_TYPE_CUSTOM,
    AQUA_MODEL_TYPE_MULTI,
    AQUA_MODEL_TYPE_SERVICE,
    AQUA_MULTI_MODEL_CONFIG,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    MODEL_NAME_DELIMITER,
    UNKNOWN_DICT,
    UNKNOWN_ENUM_VALUE,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.model import AquaModelApp
from ads.aqua.model.constants import (
    AquaModelMetadataKeys,
    ModelCustomMetadataFields,
    ModelTask,
)
from ads.aqua.model.enums import MultiModelSupportedTaskType
from ads.aqua.model.utils import (
    extract_base_model_from_ft,
    extract_fine_tune_artifacts_path,
)
from ads.aqua.modeldeployment.config_loader import (
    AquaDeploymentConfig,
    ConfigurationItem,
    ModelDeploymentConfigSummary,
    MultiModelDeploymentConfigLoader,
)
from ads.aqua.modeldeployment.constants import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIME,
    DeploymentType,
)
from ads.aqua.modeldeployment.entities import (
    AquaDeployment,
    AquaDeploymentDetail,
    ConfigValidationError,
    CreateModelDeploymentDetails,
    ModelDeploymentDetails,
    UpdateModelDeploymentDetails,
)
from ads.aqua.modeldeployment.model_group_config import ModelGroupConfig
from ads.aqua.shaperecommend.recommend import AquaShapeRecommend
from ads.aqua.shaperecommend.shape_report import (
    RequestRecommend,
    ShapeRecommendationReport,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import UNKNOWN, get_log_links
from ads.common.work_request import DataScienceWorkRequest
from ads.config import (
    AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME,
    AQUA_MODEL_DEPLOYMENT_FOLDER,
    AQUA_TELEMETRY_BUCKET,
    AQUA_TELEMETRY_BUCKET_NS,
    COMPARTMENT_OCID,
    PROJECT_OCID,
)
from ads.model.datascience_model import DataScienceModel
from ads.model.datascience_model_group import DataScienceModelGroup
from ads.model.deployment import (
    ModelDeployment,
    ModelDeploymentContainerRuntime,
    ModelDeploymentInfrastructure,
    ModelDeploymentMode,
)
from ads.model.deployment.model_deployment import ModelDeploymentUpdateType
from ads.model.model_metadata import ModelCustomMetadata, ModelCustomMetadataItem
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
    get_multimodel_deployment_config(self, model_ids: List[str],...) -> ModelDeploymentConfigSummary:
        Retrieves the deployment configuration for multiple Aqua models and calculates
        the GPU allocations for all compatible shapes.
    list_shapes(self, **kwargs) -> List[Dict]:
        Lists the valid model deployment shapes.
    recommend_shape(self, **kwargs) -> ShapeRecommendationReport:
        Generates a recommendation report or table of valid GPU deployment shapes
        for the provided model and configuration.

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
        Creates a new Aqua model deployment.\n
        For detailed information about CLI flags see: https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/cli-tips.md#create-model-deployment

        Args:
            create_deployment_details : CreateModelDeploymentDetails, optional
                An instance of CreateModelDeploymentDetails containing all required and optional
                fields for creating a model deployment via Aqua.
            kwargs:
                instance_shape (str): The instance shape used for deployment.
                display_name (str): The name of the model deployment.
                compartment_id (Optional[str]): The compartment OCID.
                project_id (Optional[str]): The project OCID.
                description (Optional[str]): The description of the deployment.
                model_id (Optional[str]): The model OCID to deploy.
                models (Optional[List[AquaMultiModelRef]]): List of models for multimodel deployment.
                instance_count (int): Number of instances used for deployment.
                log_group_id (Optional[str]): OCI logging group ID for logs.
                access_log_id (Optional[str]): OCID for access logs.
                predict_log_id (Optional[str]): OCID for prediction logs.
                bandwidth_mbps (Optional[int]): Bandwidth limit on the load balancer in Mbps.
                web_concurrency (Optional[int]): Number of worker processes/threads for handling requests.
                server_port (Optional[int]): Server port for the Docker container image.
                health_check_port (Optional[int]): Health check port for the Docker container image.
                env_var (Optional[Dict[str, str]]): Environment variables for deployment.
                container_family (Optional[str]): Image family of the model deployment container runtime.
                memory_in_gbs (Optional[float]): Memory (in GB) for the selected shape.
                ocpus (Optional[float]): OCPU count for the selected shape.
                model_file (Optional[str]): File used for model deployment.
                private_endpoint_id (Optional[str]): Private endpoint ID for model deployment.
                container_image_uri (Optional[str]): Image URI for model deployment container runtime.
                cmd_var (Optional[List[str]]): Command variables for the container runtime.
                freeform_tags (Optional[Dict]): Freeform tags for model deployment.
                defined_tags (Optional[Dict]): Defined tags for model deployment.
                deployment_type (Optional[str]): The type of model deployment.
                subnet_id (Optional[str]): The custom egress for model deployment.

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

        if not (create_deployment_details.model_id or create_deployment_details.models):
            raise AquaValueError(
                "Invalid parameters for creating a model deployment. Either `model_id` or `models` must be provided."
            )

        # Set defaults for compartment and project if not provided.
        compartment_id = create_deployment_details.compartment_id or COMPARTMENT_OCID
        project_id = create_deployment_details.project_id or PROJECT_OCID
        freeform_tags = create_deployment_details.freeform_tags
        defined_tags = create_deployment_details.defined_tags

        # validate instance shape availability in compartment
        available_shapes = [
            shape.name.lower()
            for shape in self.list_shapes(compartment_id=compartment_id)
        ]

        if create_deployment_details.instance_shape.lower() not in available_shapes:
            raise AquaValueError(
                f"Invalid Instance Shape. The selected shape '{create_deployment_details.instance_shape}' "
                f"is not supported in the {self.region} region. Please choose another shape to deploy the model."
            )

        # Get container config
        container_config = self.get_container_config()

        # Create an AquaModelApp instance once to perform the deployment creation.
        model_app = AquaModelApp()
        if (
            create_deployment_details.model_id
            or create_deployment_details.deployment_type == DeploymentType.STACKED
        ):
            model = create_deployment_details.model_id
            if not model:
                if len(create_deployment_details.models) != 1:
                    raise AquaValueError(
                        "Invalid 'models' provided. Only one base model is required for model stack deployment."
                    )
                self._validate_input_models(create_deployment_details)
                model = create_deployment_details.models[0]
            else:
                try:
                    model = create_deployment_details.validate_base_model(
                        model_id=model
                    )
                except ConfigValidationError as err:
                    raise AquaValueError(f"{err}") from err

            service_model_id = model if isinstance(model, str) else model.model_id
            logger.debug(
                f"Single model ({service_model_id}) provided. "
                "Delegating to single model creation method."
            )

            aqua_model = model_app.create(
                model=model,
                compartment_id=compartment_id,
                project_id=project_id,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
            )
            task_tag = aqua_model.freeform_tags.get(Tags.TASK, UNKNOWN)
            if (
                task_tag == ModelTask.TIME_SERIES_FORECASTING
                or task_tag == ModelTask.TIME_SERIES_FORECASTING.replace("-", "_")
            ):
                create_deployment_details.env_var.update(
                    {Tags.TASK.upper(): ModelTask.TIME_SERIES_FORECASTING}
                )
            return self._create(
                aqua_model=aqua_model,
                create_deployment_details=create_deployment_details,
                container_config=container_config,
            )
        # TODO: add multi model validation from deployment_type
        else:
            source_models, source_model_ids = self._validate_input_models(
                create_deployment_details
            )

            base_model_ids = [
                model.model_id for model in create_deployment_details.models
            ]

            try:
                model_config_summary = self.get_multimodel_deployment_config(
                    model_ids=base_model_ids, compartment_id=compartment_id
                )
                if not model_config_summary.gpu_allocation:
                    raise AquaValueError(model_config_summary.error_message)

                create_deployment_details.validate_multimodel_deployment_feasibility(
                    models_config_summary=model_config_summary
                )
            except ConfigValidationError as err:
                raise AquaValueError(f"{err}") from err

            service_inference_containers = container_config.inference.values()

            supported_container_families = [
                container_config_item.family
                for container_config_item in service_inference_containers
                if any(
                    usage.upper() in container_config_item.usages
                    for usage in [Usage.MULTI_MODEL, Usage.OTHER]
                )
            ]

            if not supported_container_families:
                raise AquaValueError(
                    "Currently, there are no containers that support multi-model deployment."
                )

            # Check if provided container family supports multi-model deployment
            if (
                create_deployment_details.container_family
                and create_deployment_details.container_family
                not in supported_container_families
            ):
                raise AquaValueError(
                    f"Unsupported deployment container '{create_deployment_details.container_family}'. "
                    f"Only {supported_container_families} families are supported for multi-model deployments."
                )

            # Verify if it matches one of the registered containers and attempt to
            # extract the container family from there.
            # If the container is not recognized, we can only issue a warning that
            # the provided container may not support multi-model deployment.
            if create_deployment_details.container_image_uri:
                selected_container_name = ContainerPath(
                    full_path=create_deployment_details.container_image_uri
                ).name

                container_config_item = next(
                    (
                        container_config_item
                        for container_config_item in service_inference_containers
                        if ContainerPath(
                            full_path=f"{container_config_item.name}:{container_config_item.version}"
                        ).name.upper()
                        == selected_container_name.upper()
                    ),
                    None,
                )

                if (
                    container_config_item
                    and container_config_item.family not in supported_container_families
                ):
                    raise AquaValueError(
                        f"Unsupported deployment container '{create_deployment_details.container_image_uri}'. "
                        f"Only {supported_container_families} families are supported for multi-model deployments."
                    )

                if not container_config_item:
                    logger.warning(
                        f"The provided container `{create_deployment_details.container_image_uri}` may not support multi-model deployment. "
                        f"Only the following container families are supported: {supported_container_families}."
                    )

            logger.debug(
                f"Multi models ({source_model_ids}) provided. Delegating to multi model creation method."
            )

            (
                model_group_display_name,
                model_group_description,
                tags,
                model_custom_metadata,
                combined_model_names,
            ) = self._build_model_group_configs(
                models=create_deployment_details.models,
                deployment_details=create_deployment_details,
                model_config_summary=model_config_summary,
                freeform_tags=freeform_tags,
                source_models=source_models,
            )

            aqua_model_group = model_app.create_multi(
                models=create_deployment_details.models,
                model_custom_metadata=model_custom_metadata,
                model_group_display_name=model_group_display_name,
                model_group_description=model_group_description,
                tags=tags,
                combined_model_names=combined_model_names,
                compartment_id=compartment_id,
                project_id=project_id,
                defined_tags=defined_tags,
            )
            return self._create_multi(
                aqua_model_group=aqua_model_group,
                create_deployment_details=create_deployment_details,
                container_config=container_config,
            )

    def _validate_input_models(
        self,
        deployment_details: ModelDeploymentDetails,
    ):
        """Validates the base models and associated fine tuned models from 'models' in create_deployment_details or update_deployment_details for stacked or multi model deployment."""
        # Collect all unique model IDs (including fine-tuned models)
        source_model_ids = list(
            {
                model_id
                for model in deployment_details.models
                for model_id in model.all_model_ids()
            }
        )
        logger.debug(
            "Fetching source model metadata for model IDs: %s", source_model_ids
        )
        # Fetch source model metadata
        source_models = self.get_multi_source(source_model_ids) or {}

        try:
            deployment_details.validate_input_models(model_details=source_models)
        except ConfigValidationError as err:
            raise AquaValueError(f"{err}") from err

        return source_models, source_model_ids

    def _build_model_group_configs(
        self,
        models: List[AquaMultiModelRef],
        deployment_details: Union[
            CreateModelDeploymentDetails, UpdateModelDeploymentDetails
        ],
        model_config_summary: ModelDeploymentConfigSummary,
        freeform_tags: Optional[Dict] = None,
        source_models: Optional[Dict[str, DataScienceModel]] = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple:
        """
        Builds configs for a multi-model grouping using the provided model list.

        Parameters
        ----------
        models : List[AquaMultiModelRef]
            List of AquaMultiModelRef instances for creating a multi-model group.
        deployment_details : Union[CreateModelDeploymentDetails, UpdateModelDeploymentDetails]
            An instance of CreateModelDeploymentDetails or UpdateModelDeploymentDetails containing all required and optional
            fields for creating or updating a model deployment via Aqua.
        model_config_summary : ModelConfigSummary
            Summary Model Deployment configuration for the group of models.
        freeform_tags : Optional[Dict]
            Freeform tags for the model.
        source_models: Optional[Dict[str, DataScienceModel]]
            A mapping of model OCIDs to their corresponding `DataScienceModel` objects.
            This dictionary contains metadata for all models involved in the multi-model deployment,
            including both base models and fine-tuned weights.

        Returns
        -------
        tuple
            A tuple of required metadata ('multi_model_metadata' and 'MULTI_MODEL_CONFIG') and strings to create model group.
        """

        if not models:
            raise AquaValueError(
                "Model list cannot be empty. Please provide at least one model for deployment."
            )

        display_name_list = []
        model_custom_metadata = ModelCustomMetadata()

        service_inference_containers = (
            self.get_container_config().to_dict().get("inference")
        )

        supported_container_families = [
            container_config_item.family
            for container_config_item in service_inference_containers
            if any(
                usage.upper() in container_config_item.usages
                for usage in [Usage.MULTI_MODEL, Usage.OTHER]
            )
        ]

        if not supported_container_families:
            raise AquaValueError(
                "Currently, there are no containers that support multi-model deployment."
            )

        selected_models_deployment_containers = set()

        if not source_models:
            # Collect all unique model IDs (including fine-tuned models)
            source_model_ids = list(
                {model_id for model in models for model_id in model.all_model_ids()}
            )
            logger.debug(
                "Fetching source model metadata for model IDs: %s", source_model_ids
            )

            # Fetch source model metadata
            source_models = self.get_multi_source(source_model_ids) or {}

        # Process each model in the input list
        for model in models:
            # Retrieve base model metadata
            source_model: DataScienceModel = source_models.get(model.model_id)
            if not source_model:
                logger.error(
                    "Failed to fetch metadata for base model ID: %s", model.model_id
                )
                raise AquaValueError(
                    f"Unable to retrieve metadata for base model ID: {model.model_id}."
                )

            # Use display name as fallback if model name not provided
            model.model_name = model.model_name or source_model.display_name

            # Validate model file description
            model_file_description = source_model.model_file_description
            if not model_file_description:
                logger.error(
                    "Model '%s' (%s) has no file description.",
                    source_model.display_name,
                    model.model_id,
                )
                raise AquaValueError(
                    f"Model '{source_model.display_name}' (ID: {model.model_id}) has no file description. "
                    "Please register the model with a file description."
                )

            # Ensure base model has a valid artifact
            if not source_model.artifact:
                logger.error(
                    "Base model '%s' (%s) has no artifact.",
                    model.model_name,
                    model.model_id,
                )
                raise AquaValueError(
                    f"Model '{model.model_name}' (ID: {model.model_id}) has no registered artifacts. "
                    "Please register the model before deployment."
                )

            # Set base model artifact path
            model.artifact_location = source_model.artifact
            logger.debug(
                "Model '%s' artifact path set to: %s",
                model.model_name,
                model.artifact_location,
            )

            display_name_list.append(model.model_name)

            # Extract model task metadata from source model
            self._extract_model_task(model, source_model)

            # Process fine-tuned weights if provided
            for ft_model in model.fine_tune_weights or []:
                fine_tune_source_model: DataScienceModel = source_models.get(
                    ft_model.model_id
                )
                if not fine_tune_source_model:
                    logger.error(
                        "Failed to fetch metadata for fine-tuned model ID: %s",
                        ft_model.model_id,
                    )
                    raise AquaValueError(
                        f"Unable to retrieve metadata for fine-tuned model ID: {ft_model.model_id}."
                    )

                # Validate model file description
                ft_model_file_description = (
                    fine_tune_source_model.model_file_description
                )
                if not ft_model_file_description:
                    logger.error(
                        "Model '%s' (%s) has no file description.",
                        fine_tune_source_model.display_name,
                        ft_model.model_id,
                    )
                    raise AquaValueError(
                        f"Model '{fine_tune_source_model.display_name}' (ID: {ft_model.model_id}) has no file description. "
                        "Please register the model with a file description."
                    )

                # Extract fine-tuned model path
                _, fine_tune_path = extract_fine_tune_artifacts_path(
                    fine_tune_source_model
                )
                logger.debug(
                    "Resolved fine-tuned model path for '%s': %s",
                    ft_model.model_id,
                    fine_tune_path,
                )
                ft_model.model_path = (
                    ft_model.model_id + "/" + fine_tune_path.lstrip("/")
                )

                # Use fallback name if needed
                ft_model.model_name = (
                    ft_model.model_name or fine_tune_source_model.display_name
                )

                display_name_list.append(ft_model.model_name)

            # Validate deployment container consistency
            deployment_container = source_model.custom_metadata_list.get(
                ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
                ModelCustomMetadataItem(
                    key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER
                ),
            ).value

            if deployment_container not in supported_container_families:
                logger.error(
                    "Unsupported deployment container '%s' for model '%s'. Supported: %s",
                    deployment_container,
                    source_model.id,
                    supported_container_families,
                )
                raise AquaValueError(
                    f"Unsupported deployment container '{deployment_container}' for model '{source_model.id}'. "
                    f"Only {supported_container_families} are supported for multi-model deployments."
                )

            selected_models_deployment_containers.add(deployment_container)

        if not selected_models_deployment_containers:
            raise AquaValueError(
                "None of the selected models are associated with a recognized container family. "
                "Please review the selected models, or select a different group of models."
            )

        # Check if the all models in the group shares same container family
        if len(selected_models_deployment_containers) > 1:
            deployment_container = get_preferred_compatible_family(
                selected_families=selected_models_deployment_containers
            )
            if not deployment_container:
                raise AquaValueError(
                    "The selected models are associated with different container families: "
                    f"{list(selected_models_deployment_containers)}."
                    "For multi-model deployment, all models in the group must belong to the same container "
                    "family or to compatible container families."
                )
        else:
            deployment_container = selected_models_deployment_containers.pop()

        # Generate model group details
        timestamp = datetime.now().strftime("%Y%m%d")
        model_group_display_name = f"model_group_{timestamp}"
        combined_model_names = ", ".join(display_name_list)
        model_group_description = f"Multi-model grouping using {combined_model_names}."

        # Add global metadata
        model_custom_metadata.add(
            key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
            value=deployment_container,
            description=f"Inference container mapping for {model_group_display_name}",
            category="Other",
        )
        model_custom_metadata.add(
            key=ModelCustomMetadataFields.MULTIMODEL_GROUP_COUNT,
            value=str(len(models)),
            description="Number of models in the group.",
            category="Other",
        )
        model_custom_metadata.add(
            key=AQUA_MULTI_MODEL_CONFIG,
            value=self._build_model_group_config(
                deployment_details=deployment_details,
                model_config_summary=model_config_summary,
                deployment_container=deployment_container,
            ).model_dump_json(),
            description="Configs required to deploy multi models.",
            category="Other",
        )
        model_custom_metadata.add(
            key=ModelCustomMetadataFields.MULTIMODEL_METADATA,
            value=json.dumps([model.model_dump() for model in models]),
            description="Metadata to store user's multi model input.",
            category="Other",
        )

        # Combine tags. The `Tags.AQUA_TAG` has been excluded, because we don't want to show
        # the models created for multi-model purpose in the AQUA models list.
        tags = {
            # Tags.AQUA_TAG: "active",
            Tags.MULTIMODEL_TYPE_TAG: "true",
            **(freeform_tags or {}),
        }

        return (
            model_group_display_name,
            model_group_description,
            tags,
            model_custom_metadata,
            combined_model_names,
        )

    def _extract_model_task(
        self,
        model: AquaMultiModelRef,
        source_model: DataScienceModel,
    ) -> None:
        """In a Multi Model Deployment, will set model_task parameter in AquaMultiModelRef from freeform tags or user"""
        # user does not supply model task, we extract from model metadata
        if not model.model_task:
            model.model_task = source_model.freeform_tags.get(Tags.TASK, UNKNOWN)

        task_tag = re.sub(r"-", "_", model.model_task).lower()
        # re-visit logic when more model task types are supported
        if task_tag in MultiModelSupportedTaskType:
            model.model_task = task_tag
        else:
            raise AquaValueError(
                f"Invalid or missing {task_tag} tag for selected model {source_model.display_name}. "
                f"Currently only `{MultiModelSupportedTaskType.values()}` models are supported for multi model deployment."
            )

    def _build_model_group_config(
        self,
        deployment_details: Union[
            CreateModelDeploymentDetails, UpdateModelDeploymentDetails
        ],
        model_config_summary,
        deployment_container: str,
    ) -> ModelGroupConfig:
        """Builds model group config required to deploy multi models."""
        container_type_key = deployment_details.container_family or deployment_container
        container_config = self.get_container_config_item(container_type_key)
        container_spec = container_config.spec if container_config else UNKNOWN

        container_params = container_spec.cli_param if container_spec else UNKNOWN

        multi_model_config = ModelGroupConfig.from_model_deployment_details(
            deployment_details,
            model_config_summary,
            container_type_key,
            container_params,
        )

        return multi_model_config

    def _create(
        self,
        aqua_model: Union[DataScienceModel, DataScienceModelGroup],
        create_deployment_details: CreateModelDeploymentDetails,
        container_config: Dict,
    ) -> AquaDeployment:
        """Builds the configurations required by single model deployment and creates the deployment.

        Parameters
        ----------
        aqua_model : DataScienceModel
            An instance of Aqua data science model.
        create_deployment_details : CreateModelDeploymentDetails
            An instance of CreateModelDeploymentDetails containing all required and optional
            fields for creating a model deployment via Aqua.
        container_config: Dict
            Container config dictionary.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        tags = {}
        for tag in [
            Tags.AQUA_SERVICE_MODEL_TAG,
            Tags.AQUA_FINE_TUNED_MODEL_TAG,
            Tags.AQUA_TAG,
            Tags.BASE_MODEL_CUSTOM,
        ]:
            if tag in aqua_model.freeform_tags:
                tags[tag] = aqua_model.freeform_tags[tag]

        tags.update({Tags.AQUA_MODEL_NAME_TAG: aqua_model.display_name})
        tags.update({Tags.TASK: aqua_model.freeform_tags.get(Tags.TASK, UNKNOWN)})

        # Set up info to get deployment config
        config_source_id = (
            create_deployment_details.model_id
            or create_deployment_details.models[0].model_id
        )
        model_name = aqua_model.display_name

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

        is_fine_tuned_model = Tags.AQUA_FINE_TUNED_MODEL_TAG in aqua_model.freeform_tags

        if is_fine_tuned_model:
            config_source_id, model_name = extract_base_model_from_ft(aqua_model)
            _, fine_tune_output_path = extract_fine_tune_artifacts_path(aqua_model)
            env_var.update({"FT_MODEL": f"{fine_tune_output_path}"})

        container_type_key = self._get_container_type_key(
            model=aqua_model,
            container_family=create_deployment_details.container_family,
        )

        container_image_uri = (
            create_deployment_details.container_image_uri
            or self.get_container_image(container_type=container_type_key)
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
            Tags.MODEL_FORMAT, ModelFormat.SAFETENSORS
        ).upper()
        model_format = model_formats_str.split(",")

        # Figure out a better way to handle this in future release
        if (
            ModelFormat.GGUF in model_format
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

        # Fetch the startup cli command for the container
        # container_index.json will have "containerSpec" section which will provide the cli params for
        # a given container family
        container_config = self.get_container_config_item(container_type_key)

        container_spec = container_config.spec if container_config else UNKNOWN
        # these params cannot be overridden for Aqua deployments
        params = container_spec.cli_param if container_spec else UNKNOWN
        server_port = create_deployment_details.server_port or (
            container_spec.server_port if container_spec else None
        )
        # Give precendece to the input parameter
        health_check_port = create_deployment_details.health_check_port or (
            container_spec.health_check_port if container_spec else None
        )

        deployment_config = self.get_deployment_config(model_id=config_source_id)

        # Loads frameworks specific default params from the configuration
        config_params = deployment_config.configuration.get(
            create_deployment_details.instance_shape, ConfigurationItem()
        ).parameters.get(get_container_params_type(container_type_key), UNKNOWN)

        # Loads default environment variables from the configuration
        config_env = deployment_config.configuration.get(
            create_deployment_details.instance_shape, ConfigurationItem()
        ).env.get(get_container_params_type(container_type_key), {})

        # Merges user provided environment variables with the ones provided in the deployment config
        # The values provided by user will override the ones provided by default config
        env_var = {**config_env, **env_var}

        # SMM Parameter Resolution Logic
        # Check the raw user input from create_deployment_details to determine intent.
        # We cannot use the merged 'env_var' here because it may already contain defaults.
        user_input_env = create_deployment_details.env_var or {}
        user_input_params = user_input_env.get("PARAMS")

        deployment_params = ""

        if user_input_params is None:
            # Case 1: None (CLI default) -> Load full defaults from config
            logger.info("No PARAMS provided (None). Loading default SMM parameters.")
            deployment_params = config_params
        elif str(user_input_params).strip() == "":
            # Case 2: Empty String (UI Clear) -> Explicitly use no parameters
            logger.info("Empty PARAMS provided. Clearing all parameters.")
            deployment_params = ""
        else:
            # Case 3: Value Provided -> Use exact user value (No merging)
            logger.info(
                f"User provided PARAMS. Using exact user values: {user_input_params}"
            )
            deployment_params = user_input_params

        # Validate the resolved parameters
        if deployment_params:
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

            restricted_params = find_restricted_params(
                params, deployment_params, container_type_key
            )
            if restricted_params:
                raise AquaValueError(
                    f"Parameters {restricted_params} are set by Aqua "
                    f"and cannot be overridden or are invalid."
                )

        params = f"{params} {deployment_params}".strip()

        if isinstance(aqua_model, DataScienceModelGroup):
            tags.update({Tags.STACKED_MODEL_TYPE_TAG: "true"})
            env_var.update({"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"})
            env_var.update(
                {"MODEL": f"{AQUA_MODEL_DEPLOYMENT_FOLDER}{aqua_model.base_model_id}/"}
            )

            base_model_inference_key = aqua_model.base_model_id
            for item in aqua_model.member_models:
                if item["model_id"] == aqua_model.base_model_id:
                    if item["inference_key"]:
                        base_model_inference_key = item["inference_key"]
                    break

            params_dict = get_params_dict(params)
            # updates `--served-model-name` with service model inference key
            params_dict.update({"--served-model-name": base_model_inference_key})
            # TODO: sets `--max-lora-rank` as 32 in params for now, will revisit later
            params_dict.update({"--max-lora-rank": 32})
            # adds `--enable_lora` to parameters
            params_dict.update({"--enable_lora": UNKNOWN})
            params = build_params_string(params_dict)
        elif create_deployment_details.model_name and "--served-model-name" in params:
            # Replace existing --served-model-name argument with custom name provided by user
            params = re.sub(
                r"--served-model-name\s+\S+",
                f"--served-model-name {create_deployment_details.model_name}",
                params,
            )

        if params:
            env_var.update({"PARAMS": params})
        env_vars = container_spec.env_vars if container_spec else []
        for env in env_vars:
            if isinstance(env, dict):
                env = {k: v for k, v in env.items() if v}
                for key, _ in env.items():
                    if key not in env_var:
                        env_var.update(env)

        env_var.update({"AQUA_TELEMETRY_BUCKET_NS": AQUA_TELEMETRY_BUCKET_NS})
        env_var.update({"AQUA_TELEMETRY_BUCKET": AQUA_TELEMETRY_BUCKET})

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
        aqua_model_group: DataScienceModelGroup,
        create_deployment_details: CreateModelDeploymentDetails,
        container_config: AquaContainerConfig,
    ) -> AquaDeployment:
        """Builds the environment variables required by multi deployment container and creates the deployment.

        Parameters
        ----------
        aqua_model_group : DataScienceModelGroup
            An instance of Aqua data science model group.
        create_deployment_details : CreateModelDeploymentDetails
            An instance of CreateModelDeploymentDetails containing all required and optional
            fields for creating a model deployment via Aqua.
        container_config: Dict
            Container config dictionary.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        model_name_list = []
        env_var = {**(create_deployment_details.env_var or UNKNOWN_DICT)}

        container_type_key = self._get_container_type_key(
            model=aqua_model_group,
            container_family=create_deployment_details.container_family,
        )
        container_config = self.get_container_config_item(container_type_key)
        container_spec = container_config.spec if container_config else UNKNOWN

        container_spec_env_vars = container_spec.env_vars if container_spec else []
        for env in container_spec_env_vars:
            if isinstance(env, dict):
                env = {k: v for k, v in env.items() if v}
                for key, _ in env.items():
                    if key not in env_var:
                        env_var.update(env)

        logger.info(f"Env vars used for deploying {aqua_model_group.id} : {env_var}.")

        container_image_uri = (
            create_deployment_details.container_image_uri
            or self.get_container_image(container_type=container_type_key)
        )
        server_port = create_deployment_details.server_port or (
            container_spec.server_port if container_spec else None
        )
        health_check_port = create_deployment_details.health_check_port or (
            container_spec.health_check_port if container_spec else None
        )
        tags = {
            Tags.AQUA_MODEL_ID_TAG: aqua_model_group.id,
            Tags.MULTIMODEL_TYPE_TAG: "true",
            Tags.AQUA_TAG: "active",
            **(create_deployment_details.freeform_tags or UNKNOWN_DICT),
        }

        model_name = f"{MODEL_NAME_DELIMITER} ".join(model_name_list)

        aqua_deployment = self._create_deployment(
            create_deployment_details=create_deployment_details,
            aqua_model_id=aqua_model_group.id,
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
            .with_subnet_id(create_deployment_details.subnet_id)
            .with_capacity_reservation_ids(
                create_deployment_details.capacity_reservation_ids or []
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
            .with_region(self.region)
            .with_overwrite_existing_artifact(True)
            .with_remove_existing_artifact(True)
        )
        if self._if_model_group(aqua_model_id):
            container_runtime.with_model_group_id(aqua_model_id)
        else:
            container_runtime.with_model_uri(aqua_model_id)
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
            f"Aqua model deployment {deployment_id} created for model {aqua_model_id}. Work request Id is {deployment.dsc_model_deployment.workflow_req_id}"
        )

        progress_thread = threading.Thread(
            target=self.get_deployment_status,
            args=(
                deployment,
                deployment.dsc_model_deployment.workflow_req_id,
                model_type,
                model_name,
            ),
            daemon=True,
        )
        progress_thread.start()

        # we arbitrarily choose last 8 characters of OCID to identify MD in telemetry
        deployment_short_ocid = get_ocid_substring(deployment_id, key_len=8)

        # Prepare telemetry kwargs
        telemetry_kwargs = {"ocid": deployment_short_ocid}

        if Tags.BASE_MODEL_CUSTOM in tags:
            telemetry_kwargs["custom_base_model"] = True

        if Tags.MULTIMODEL_TYPE_TAG in tags:
            telemetry_kwargs["deployment_type"] = DeploymentType.MULTI
        elif Tags.STACKED_MODEL_TYPE_TAG in tags:
            telemetry_kwargs["deployment_type"] = DeploymentType.STACKED
        else:
            telemetry_kwargs["deployment_type"] = DeploymentType.SINGLE

        telemetry_kwargs["container"] = (
            create_deployment_details.container_family
            or create_deployment_details.container_image_uri
        )

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
            **{"ocid": deployment_short_ocid},
        )

        return AquaDeployment.from_oci_model_deployment(
            deployment.dsc_model_deployment, self.region
        )

    @staticmethod
    def _get_container_type_key(
        model: Union[DataScienceModel, DataScienceModelGroup], container_family: str
    ) -> str:
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

    @telemetry(entry_point="plugin=deployment&action=update", name="aqua")
    def update(
        self,
        model_deployment_id: str,
        update_model_deployment_details: Optional[UpdateModelDeploymentDetails] = None,
        **kwargs,
    ) -> AquaDeployment:
        """Updates a AQUA model group deployment.

        Args:
            update_model_deployment_details : UpdateModelDeploymentDetails, optional
                An instance of UpdateModelDeploymentDetails containing all optional
                fields for updating a model deployment via Aqua.
            kwargs:
                display_name (str): The name of the model deployment.
                description (Optional[str]): The description of the deployment.
                models (Optional[List[AquaMultiModelRef]]): List of models for deployment.
                instance_count (int): Number of instances used for deployment.
                log_group_id (Optional[str]): OCI logging group ID for logs.
                access_log_id (Optional[str]): OCID for access logs.
                predict_log_id (Optional[str]): OCID for prediction logs.
                bandwidth_mbps (Optional[int]): Bandwidth limit on the load balancer in Mbps.
                web_concurrency (Optional[int]): Number of worker processes/threads for handling requests.
                memory_in_gbs (Optional[float]): Memory (in GB) for the selected shape.
                ocpus (Optional[float]): OCPU count for the selected shape.
                freeform_tags (Optional[Dict]): Freeform tags for model deployment.
                defined_tags (Optional[Dict]): Defined tags for model deployment.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        if not update_model_deployment_details:
            try:
                update_model_deployment_details = UpdateModelDeploymentDetails(**kwargs)
            except ValidationError as ex:
                custom_errors = build_pydantic_error_message(ex)
                raise AquaValueError(
                    f"Invalid parameters for updating a model group deployment. Error details: {custom_errors}."
                ) from ex

        model_deployment = ModelDeployment.from_id(model_deployment_id)

        infrastructure = model_deployment.infrastructure
        runtime = model_deployment.runtime

        if not runtime.model_group_id:
            raise AquaValueError(
                "Invalid 'model_deployment_id'. Only model group deployment is supported to update."
            )

        # updates model group if fine tuned weights changed.
        model = self._update_model_group(
            runtime.model_group_id, update_model_deployment_details, model_deployment
        )

        # updates model group deployment infrastructure
        (
            infrastructure.with_bandwidth_mbps(
                update_model_deployment_details.bandwidth_mbps
                or infrastructure.bandwidth_mbps
            )
            .with_replica(
                update_model_deployment_details.instance_count or infrastructure.replica
            )
            .with_web_concurrency(
                update_model_deployment_details.web_concurrency
                or infrastructure.web_concurrency
            )
        )

        if (
            update_model_deployment_details.log_group_id
            and update_model_deployment_details.access_log_id
        ):
            infrastructure.with_access_log(
                log_group_id=update_model_deployment_details.log_group_id,
                log_id=update_model_deployment_details.access_log_id,
            )

        if (
            update_model_deployment_details.log_group_id
            and update_model_deployment_details.predict_log_id
        ):
            infrastructure.with_predict_log(
                log_group_id=update_model_deployment_details.log_group_id,
                log_id=update_model_deployment_details.predict_log_id,
            )

        if (
            update_model_deployment_details.memory_in_gbs
            and update_model_deployment_details.ocpus
            and infrastructure.shape_name.endswith("Flex")
        ):
            infrastructure.with_shape_config_details(
                ocpus=update_model_deployment_details.ocpus,
                memory_in_gbs=update_model_deployment_details.memory_in_gbs,
            )

        # applies ZDT as default type to update parameters if model group id hasn't been changed
        update_type = ModelDeploymentUpdateType.ZDT
        # applies LIVE update if model group id has been changed
        if runtime.model_group_id != model.id:
            runtime.with_model_group_id(model.id)
            if model.dsc_model_group.model_group_details.type == DeploymentType.STACKED:
                # only applies LIVE update for stacked deployment
                update_type = ModelDeploymentUpdateType.LIVE

        freeform_tags = (
            update_model_deployment_details.freeform_tags
            or model_deployment.freeform_tags
        )
        defined_tags = (
            update_model_deployment_details.defined_tags
            or model_deployment.defined_tags
        )

        # updates model group deployment
        (
            model_deployment.with_display_name(
                update_model_deployment_details.display_name
                or model_deployment.display_name
            )
            .with_description(
                update_model_deployment_details.description
                or model_deployment.description
            )
            .with_freeform_tags(**(freeform_tags or {}))
            .with_defined_tags(**(defined_tags or {}))
            .with_infrastructure(infrastructure)
            .with_runtime(runtime)
        )

        model_deployment.update(wait_for_completion=False, update_type=update_type)

        logger.info(f"Updating Aqua Model Deployment {model_deployment.id}.")

        return AquaDeployment.from_oci_model_deployment(
            model_deployment.dsc_model_deployment, self.region
        )

    def _update_model_group(
        self,
        model_group_id: str,
        update_model_deployment_details: UpdateModelDeploymentDetails,
        model_deployment: ModelDeployment,
    ) -> DataScienceModelGroup:
        """Creates a new model group if fine tuned weights changed.

        Parameters
        ----------
        model_group_id: str
            The model group id.
        update_model_deployment_details: UpdateModelDeploymentDetails
            An instance of UpdateModelDeploymentDetails containing all optional
            fields for updating a model deployment via Aqua.
        model_deployment: ModelDeployment
            An instance of ModelDeployment.

        Returns
        -------
        DataScienceModelGroup
            The instance of DataScienceModelGroup.
        """
        model_group = DataScienceModelGroup.from_id(model_group_id)
        if update_model_deployment_details.models:
            # validates input base and fine tune models
            source_models, _ = self._validate_input_models(
                update_model_deployment_details
            )
            if (
                model_group.dsc_model_group.model_group_details.type
                == DeploymentType.STACKED
            ):
                # create a new model group if fine tune weights changed as member models in ds model group is inmutable
                if len(update_model_deployment_details.models) != 1:
                    raise AquaValueError(
                        "Invalid 'models' provided. Only one base model is required for updating model stack deployment."
                    )
                target_stacked_model = update_model_deployment_details.models[0]
                target_base_model_id = target_stacked_model.model_id
                if model_group.base_model_id != target_base_model_id:
                    raise AquaValueError(
                        "Invalid parameter 'models'. Base model id can't be changed for stacked model deployment."
                    )

                # add member models
                member_models = [
                    {
                        "inference_key": fine_tune_weight.model_name,
                        "model_id": fine_tune_weight.model_id,
                    }
                    for fine_tune_weight in target_stacked_model.fine_tune_weights
                ]
                # add base model
                member_models.append(
                    {
                        "inference_key": target_stacked_model.model_name,
                        "model_id": target_base_model_id,
                    }
                )

                # creates a model group with the same configurations from original model group except member models
                model_group = (
                    DataScienceModelGroup()
                    .with_compartment_id(model_group.compartment_id)
                    .with_project_id(model_group.project_id)
                    .with_display_name(model_group.display_name)
                    .with_description(model_group.description)
                    .with_freeform_tags(**(model_group.freeform_tags or {}))
                    .with_defined_tags(**(model_group.defined_tags or {}))
                    .with_custom_metadata_list(model_group.custom_metadata_list)
                    .with_base_model_id(target_base_model_id)
                    .with_member_models(member_models)
                    .create()
                )

                logger.info(
                    f"Model group of base model {target_base_model_id} has been updated: {model_group.id}."
                )
            else:
                compartment_id = model_deployment.infrastructure.compartment_id
                project_id = model_deployment.infrastructure.project_id
                freeform_tags = (
                    update_model_deployment_details.freeform_tags
                    or model_deployment.freeform_tags
                )
                defined_tags = (
                    update_model_deployment_details.defined_tags
                    or model_deployment.defined_tags
                )
                # needs instance shape here for building the multi model config from update_model_deployment_details
                update_model_deployment_details.instance_shape = (
                    model_deployment.infrastructure.shape_name
                )

                # rebuilds MULTI_MODEL_CONFIG and creates model group
                base_model_ids = [
                    model.model_id for model in update_model_deployment_details.models
                ]

                try:
                    model_config_summary = self.get_multimodel_deployment_config(
                        model_ids=base_model_ids, compartment_id=compartment_id
                    )
                    if not model_config_summary.gpu_allocation:
                        raise AquaValueError(model_config_summary.error_message)

                    update_model_deployment_details.validate_multimodel_deployment_feasibility(
                        models_config_summary=model_config_summary
                    )
                except ConfigValidationError as err:
                    raise AquaValueError(f"{err}") from err

                (
                    model_group_display_name,
                    model_group_description,
                    tags,
                    model_custom_metadata,
                    combined_model_names,
                ) = self._build_model_group_configs(
                    models=update_model_deployment_details.models,
                    deployment_details=update_model_deployment_details,
                    model_config_summary=model_config_summary,
                    freeform_tags=freeform_tags,
                    source_models=source_models,
                )

                model_group = AquaModelApp().create_multi(
                    models=update_model_deployment_details.models,
                    model_custom_metadata=model_custom_metadata,
                    model_group_display_name=model_group_display_name,
                    model_group_description=model_group_description,
                    tags=tags,
                    combined_model_names=combined_model_names,
                    compartment_id=compartment_id,
                    project_id=project_id,
                    defined_tags=defined_tags,
                )

                logger.info(
                    f"Model group of multi model deployment {model_deployment.id} has been updated: {model_group.id}."
                )

        return model_group

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
            # skipping the AQUA model deployments that are created with UNKNOWN deployment type
            if (
                model_deployment.model_deployment_configuration_details.deployment_type
                in [UNKNOWN_ENUM_VALUE]
            ):
                logger.debug(
                    f"Skipping model deployment with UNKNOWN deployment type: "
                    f"{getattr(model_deployment, 'id', '<missing_id>')}"
                )
                continue

            oci_aqua = (
                (
                    Tags.AQUA_TAG in model_deployment.freeform_tags
                    or Tags.AQUA_TAG.lower() in model_deployment.freeform_tags
                )
                if model_deployment.freeform_tags
                else False
            )

            if oci_aqua:
                try:
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
                except Exception as e:
                    logger.error(
                        (
                            f"Failed to process AQUA model deployment "
                            f"'{getattr(model_deployment, 'display_name', '<unknown>')}' "
                            f"(OCID: {getattr(model_deployment, 'id', '<missing>')}, Region: {self.region}).\n"
                            f"Reason: {type(e).__name__}: {e}"
                        ),
                        exc_info=True,
                    )
                    # raise AquaRuntimeError(
                    #     f"There was an issue processing the list of model deployments . Error: {str(e)}"
                    # ) from e

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

        aqua_deployment = AquaDeployment.from_oci_model_deployment(
            model_deployment, self.region
        )
        if Tags.MULTIMODEL_TYPE_TAG in model_deployment.freeform_tags:
            aqua_model_id = model_deployment.freeform_tags.get(
                Tags.AQUA_MODEL_ID_TAG, UNKNOWN
            )
            if not aqua_model_id:
                raise AquaRuntimeError(
                    f"Invalid multi model deployment {model_deployment_id}."
                    f"Make sure the {Tags.AQUA_MODEL_ID_TAG} tag is added to the deployment."
                )

            if self._if_model_group(aqua_model_id):
                aqua_model = DataScienceModelGroup.from_id(aqua_model_id)
            else:
                aqua_model = DataScienceModel.from_id(aqua_model_id)

            custom_metadata_list = aqua_model.custom_metadata_list
            multi_model_metadata_value = custom_metadata_list.get(
                ModelCustomMetadataFields.MULTIMODEL_METADATA,
                ModelCustomMetadataItem(
                    key=ModelCustomMetadataFields.MULTIMODEL_METADATA
                ),
            ).value
            if not multi_model_metadata_value:
                raise AquaRuntimeError(
                    f"Invalid multi-model deployment: {model_deployment_id}. "
                    f"Ensure that the required custom metadata `{ModelCustomMetadataFields.MULTIMODEL_METADATA}` is added to the AQUA multi-model `{aqua_model.display_name}` ({aqua_model.id})."
                )
            multi_model_metadata = json.loads(
                multi_model_metadata_value
                if isinstance(aqua_model, DataScienceModelGroup)
                else aqua_model.dsc_model.get_custom_metadata_artifact(
                    metadata_key_name=ModelCustomMetadataFields.MULTIMODEL_METADATA
                ).decode("utf-8")
            )
            aqua_deployment.models = [
                AquaMultiModelRef(**metadata) for metadata in multi_model_metadata
            ]
        return AquaDeploymentDetail(
            **vars(aqua_deployment),
            log_group=AquaResourceIdentifier(
                log_group_id, log_group_name, log_group_url
            ),
            log=AquaResourceIdentifier(log_id, log_name, log_url),
        )

    @staticmethod
    def _if_model_group(model_id: str) -> bool:
        """Checks if it's model group id or not."""
        return "datasciencemodelgroup" in model_id.lower()

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
        config = self.get_config_from_metadata(
            model_id, AquaModelMetadataKeys.DEPLOYMENT_CONFIGURATION
        ).config

        if config:
            logger.info(
                f"Fetched {AquaModelMetadataKeys.DEPLOYMENT_CONFIGURATION} from defined metadata for model: {model_id}."
            )
            return AquaDeploymentConfig(**(config or UNKNOWN_DICT))
        config = self.get_config(
            model_id,
            DEFINED_METADATA_TO_FILE_MAP.get(
                AquaModelMetadataKeys.DEPLOYMENT_CONFIGURATION.lower()
            ),
        ).config
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
        self,
        model_ids: List[str],
        primary_model_id: Optional[str] = None,
        **kwargs: Dict,
    ) -> ModelDeploymentConfigSummary:
        """
        Retrieves the deployment configuration for multiple models and calculates
        GPU allocations across all compatible shapes.

        More details:
        https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/multimodel-deployment-tips.md#get_multimodel_deployment_config

        CLI example:
        ads aqua deployment get_multimodel_deployment_config --model_ids '["md_ocid1","md_ocid2"]'

        If a primary model ID is provided, GPU allocation will prioritize that model
        when selecting compatible shapes.

        Example:
        Assume all three models: A, B, and C, support the same shape: "BM.GPU.H100.8" and each supports the following GPU counts for that shape: 1, 2, 4, 8.
        If `no` primary model is specified, valid allocations could be: [2, 4, 2], [2, 2, 4], or [4, 2, 2]
        If `B` is set as the primary model, the allocation will be: [2, 4, 2], where B receives the maximum available GPU count

        Parameters
        ----------
        model_ids : List[str]
            A list of OCIDs for the Aqua models.
        primary_model_id : Optional[str]
            The OCID of the primary Aqua model. If provided, GPU allocation will prioritize
            this model. Otherwise, GPUs will be evenly allocated.
        **kwargs: Dict
            - compartment_id: str
                The compartment OCID to retrieve the model deployment shapes.

        Returns
        -------
        ModelDeploymentConfigSummary
            A summary of the model deployment configurations and GPU allocations.
        """
        if not model_ids:
            raise AquaValueError(
                "Model IDs were not provided. Please provide a valid list of model IDs to retrieve the multi-model deployment configuration."
            )

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        # Get the all model deployment available shapes in a given compartment
        available_shapes = self.list_shapes(compartment_id=compartment_id)

        return MultiModelDeploymentConfigLoader(
            deployment_app=self,
        ).load(
            shapes=available_shapes,
            model_ids=model_ids,
            primary_model_id=primary_model_id,
        )

    def get_deployment_default_params(
        self,
        model_id: str,
        instance_shape: str,
        gpu_count: int = None,
    ) -> Dict:
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
        default_envs = {}
        config_params = {}
        model = DataScienceModel.from_id(model_id)
        try:
            container_type_key = model.custom_metadata_list.get(
                AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME
            ).value
        except ValueError:
            container_type_key = UNKNOWN
            logger.debug(
                f"{AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME} key is not available in the "
                f"custom metadata field for model {model_id}."
            )

        if container_type_key:
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
                        default_envs = instance_shape_config.env.get(
                            get_container_env_type(container_type_key), {}
                        )
                        break

            else:
                config_params = instance_shape_config.parameters.get(
                    get_container_params_type(container_type_key), UNKNOWN
                )
                default_envs = instance_shape_config.env.get(
                    get_container_env_type(container_type_key), {}
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

        return {"data": default_params, "env": default_envs}

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

            container_config = self.get_container_config_item(container_type_key)
            container_spec = container_config.spec if container_config else UNKNOWN
            cli_params = container_spec.cli_param if container_spec else UNKNOWN

            restricted_params = find_restricted_params(
                cli_params, params, container_type_key
            )

        if restricted_params:
            raise AquaValueError(
                f"Parameters {restricted_params} are set by Aqua "
                f"and cannot be overridden or are invalid."
            )
        return {"valid": True}

    @cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=1), timer=datetime.now))
    def recommend_shape(self, **kwargs) -> Union[Table, ShapeRecommendationReport]:
        """
        Generates a recommendation report or table of valid GPU deployment shapes
        for the provided model and configuration.

        For CLI (default `generate_table=True`): generates a rich table.
        For API (`generate_table=False`): returns a structured JSON report.
        Example: ads aqua deployment recommend_shape --model-id meta-llama/Llama-3.3-70B-Instruct --generate_table false

        Args:
            model_id : str
                (Required) The OCID or Hugging Face model ID to recommend compute shapes for.
            generate_table : bool, optional
                If True, generates and returns a table (default: False).

        Returns
        -------
        Table
            If `generate_table=True`, returns a table of shape recommendations.

        ShapeRecommendationReport
            If `generate_table=False`, returns a structured recommendation report.

        Raises
        ------
        AquaValueError
            If required parameters are missing or invalid.
        """
        model_id = kwargs.pop("model_id", None)
        if not model_id:
            raise AquaValueError(
                "The 'model_id' parameter is required to generate shape recommendations. "
                "Please provide a valid OCID or Hugging Face model identifier."
            )

        logger.info(f"Starting shape recommendation for model_id: {model_id}")

        self.telemetry.record_event_async(
            category="aqua/deployment",
            action="recommend_shape",
            detail=(
                get_ocid_substring(model_id, key_len=8)
                if is_valid_ocid(ocid=model_id)
                else model_id
            ),
            **kwargs,
        )

        if is_valid_ocid(ocid=model_id):
            logger.debug(
                f"Attempting to retrieve deployment configuration for model_id={model_id}"
            )
            try:
                deployment_config = self.get_deployment_config(model_id=model_id)
                kwargs["deployment_config"] = deployment_config
                logger.debug(
                    f"Retrieved deployment configuration for model: {model_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve deployment configuration for model_id={model_id}: {e}"
                )

        try:
            request = RequestRecommend(model_id=model_id, **kwargs)
        except ValidationError as e:
            custom_error = build_pydantic_error_message(e)
            logger.error(
                f"Validation failed for shape recommendation request: {custom_error}"
            )
            raise AquaValueError(
                f"Invalid input parameters for shape recommendation: {custom_error}"
            ) from e

        try:
            shape_recommend = AquaShapeRecommend()
            logger.info(
                f"Running shape recommendation for model '{model_id}' "
                f"with generate_table={getattr(request, 'generate_table', False)}"
            )
            shape_recommend_report = shape_recommend.which_shapes(request)
            logger.info(f"Shape recommendation completed successfully for {model_id}")
            return shape_recommend_report
        except AquaValueError:
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error while generating shape recommendations: {e}"
            )
            raise AquaValueError(
                f"An unexpected error occurred during shape recommendation: {e}"
            ) from e

    @telemetry(entry_point="plugin=deployment&action=list_shapes", name="aqua")
    @cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=5), timer=datetime.now))
    def list_shapes(self, **kwargs) -> List[ComputeShapeSummary]:
        """Lists the valid model deployment shapes.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id
            for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

        Returns
        -------
        List[ComputeShapeSummary]:
            The list of the model deployment shapes.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        oci_shapes: list[ModelDeploymentShapeSummary] = self.list_resource(
            self.ds_client.list_model_deployment_shapes,
            compartment_id=compartment_id,
            **kwargs,
        )

        gpu_specs = load_gpu_shapes_index()

        return [
            ComputeShapeSummary(
                core_count=oci_shape.core_count,
                memory_in_gbs=oci_shape.memory_in_gbs,
                shape_series=oci_shape.shape_series,
                name=oci_shape.name,
                gpu_specs=gpu_specs.shapes.get(oci_shape.name)
                or gpu_specs.shapes.get(oci_shape.name.upper()),
            )
            for oci_shape in oci_shapes
        ]

    def get_deployment_status(
        self,
        deployment: ModelDeployment,
        work_request_id: str,
        model_type: str,
        model_name: str,
    ) -> None:
        """Waits for the data science  model deployment to be completed and log its status in telemetry.

        Parameters
        ----------

        model_deployment_id: str
            The id of the deployed aqua model.
        work_request_id: str
            The work request Id of the model deployment.
        model_type: str
            The type of aqua model to be deployed. Allowed values are: `custom`, `service` and `multi_model`.

        Returns
        -------
        AquaDeployment
            An Aqua deployment instance.
        """
        ocid = get_ocid_substring(deployment.id, key_len=8)
        data_science_work_request: DataScienceWorkRequest = DataScienceWorkRequest(
            work_request_id
        )
        try:
            data_science_work_request.wait_work_request(
                progress_bar_description="Creating model deployment",
                max_wait_time=DEFAULT_WAIT_TIME,
                poll_interval=DEFAULT_POLL_INTERVAL,
            )
        except Exception:
            if data_science_work_request._error_message:
                error_str = ""
                for error in data_science_work_request._error_message:
                    error_str = error_str + " " + error.message

                error_str = re.sub(r"[^a-zA-Z0-9]", " ", error_str)

                telemetry_kwargs = {
                    "ocid": ocid,
                    "model_name": model_name,
                    "work_request_error": error_str,
                }

                self.telemetry.record_event(
                    category=f"aqua/{model_type}/deployment/status",
                    action="FAILED",
                    **telemetry_kwargs,
                )
        else:
            telemetry_kwargs = {"ocid": ocid, "model_name": model_name}
            self.telemetry.record_event(
                category=f"aqua/{model_type}/deployment/status",
                action="SUCCEEDED",
                **telemetry_kwargs,
            )
