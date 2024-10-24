#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import inspect
import os
import shutil
import tempfile
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import warnings

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image

from ads.common import auth as authutil
from ads.common import logger, utils
from ads.common.decorator.utils import class_or_instance_method
from ads.common.utils import DATA_SCHEMA_MAX_COL_NUM, get_files
from ads.common.object_storage_details import ObjectStorageDetails
from ads.config import (
    CONDA_BUCKET_NS,
    JOB_RUN_COMPARTMENT_OCID,
    JOB_RUN_OCID,
    NB_SESSION_COMPARTMENT_OCID,
    NB_SESSION_OCID,
    PIPELINE_RUN_COMPARTMENT_OCID,
    PROJECT_OCID,
    TMPDIR,
)
from ads.evaluations import EvaluatorMixin
from ads.feature_engineering import ADSImage
from ads.feature_engineering.schema import Schema
from ads.feature_store.model_details import ModelDetails
from ads.model.artifact import ModelArtifact
from ads.model.common.utils import (
    _extract_locals,
    _is_json_serializable,
    fetch_manifest_from_conda_location,
    zip_artifact,
)
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIME,
    ModelDeployment,
    ModelDeploymentMode,
    ModelDeploymentProperties,
    ModelDeploymentCondaRuntime,
    ModelDeploymentInfrastructure,
    ModelDeploymentContainerRuntime,
)
from ads.model.deployment.common.utils import State as ModelDeploymentState
from ads.model.deployment.common.utils import send_request
from ads.model.model_introspect import (
    TEST_STATUS,
    Introspectable,
    IntrospectionNotPassed,
    ModelIntrospect,
)
from ads.model.model_metadata import (
    ExtendedEnumMeta,
    Framework,
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
    MetadataCustomCategory,
)
from ads.model.model_metadata_mixin import MetadataMixin
from ads.model.model_properties import ModelProperties
from ads.model.model_version_set import ModelVersionSet, _extract_model_version_set_id
from ads.model.runtime.env_info import DEFAULT_CONDA_BUCKET_NAME
from ads.model.runtime.runtime_info import RuntimeInfo
from ads.model.serde.common import SERDE
from ads.model.serde.model_input import (
    SUPPORTED_MODEL_INPUT_SERIALIZERS,
    ModelInputSerializerFactory,
    ModelInputSerializerType,
)
from ads.model.serde.model_serializer import (
    SUPPORTED_MODEL_SERIALIZERS,
    ModelSerializerFactory,
    ModelSerializerType,
)
from ads.model.transformer.onnx_transformer import ONNXTransformer

_TRAINING_RESOURCE_ID = JOB_RUN_OCID or NB_SESSION_OCID
_COMPARTMENT_OCID = (
    NB_SESSION_COMPARTMENT_OCID
    or JOB_RUN_COMPARTMENT_OCID
    or PIPELINE_RUN_COMPARTMENT_OCID
)

MODEL_DEPLOYMENT_INSTANCE_SHAPE = "VM.Standard.E4.Flex"
MODEL_DEPLOYMENT_INSTANCE_OCPUS = 1
MODEL_DEPLOYMENT_INSTANCE_MEMORY_IN_GBS = 16
MODEL_DEPLOYMENT_INSTANCE_COUNT = 1
MODEL_DEPLOYMENT_BANDWIDTH_MBPS = 10


DEFAULT_MODEL_FOLDER_NAME = "model"

ONNX_DATA_TRANSFORMER = "onnx_data_transformer.json"
_ATTRIBUTES_TO_SHOW_ = [
    "artifact_dir",
    "framework",
    "algorithm",
    "model_id",
    "model_deployment_id",
]
FRAMEWORKS_WITHOUT_ONNX_DATA_TRANSFORM = [
    Framework.TENSORFLOW,
    Framework.PYTORCH,
    Framework.SPARK,
]

VERIFY_STATUS_NAME = "verify()"
PREPARE_STATUS_NAME = "prepare()"
INITIATE_STATUS_NAME = "initiate"
SAVE_STATUS_NAME = "save()"
DEPLOY_STATUS_NAME = "deploy()"
PREDICT_STATUS_NAME = "predict()"

INITIATE_STATUS_DETAIL = "Initiated the model"
PREPARE_STATUS_GEN_RUNTIME_DETAIL = "Generated runtime.yaml"
PREPARE_STATUS_GEN_SCORE_DETAIL = "Generated score.py"
PREPARE_STATUS_SERIALIZE_MODEL_DETAIL = "Serialized model"
PREPARE_STATUS_POPULATE_METADATA_DETAIL = (
    "Populated metadata(Custom, Taxonomy and Provenance)"
)
VERIFY_STATUS_LOCAL_TEST_DETAIL = "Local tested .predict from score.py"
SAVE_STATUS_INTROSPECT_TEST_DETAIL = "Conducted Introspect Test"
SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL = "Uploaded artifact to model catalog"
DEPLOY_STATUS_DETAIL = "Deployed the model"
PREDICT_STATUS_CALL_ENDPOINT_DETAIL = "Called deployment predict endpoint"

Self = TypeVar("Self", bound="GenericModel")


class ModelDeploymentRuntimeType:
    CONDA = "conda"
    CONTAINER = "container"


class DataScienceModelType(str, metaclass=ExtendedEnumMeta):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"


class NotActiveDeploymentError(Exception):  # pragma: no cover
    def __init__(self, state: str):
        msg = (
            "To perform a prediction the deployed model needs to be in an active state. "
            f"The current state is: {state}."
        )
        super().__init__(msg)


class ArtifactsNotAvailableError(Exception):
    def __init__(
        self, msg="Model artifacts are either not generated or not available locally."
    ):
        super().__init__(msg)


class SerializeModelNotImplementedError(NotImplementedError):  # pragma: no cover
    pass


class SerializeInputNotImplementedError(NotImplementedError):  # pragma: no cover
    pass


class RuntimeInfoInconsistencyError(Exception):  # pragma: no cover
    pass


def _prepare_artifact_dir(artifact_dir: str = None) -> str:
    """Prepares artifact dir for the model.

    Parameters
    ----------
    artifact_dir: (str, optional). Defaults to `None`.
        The artifact dir that needs to be normalized.

    Returns
    -------
    str
        The artifact dir.
    """
    if artifact_dir and ObjectStorageDetails.is_oci_path(artifact_dir):
        return artifact_dir

    if artifact_dir and isinstance(artifact_dir, str):
        return os.path.abspath(os.path.expanduser(artifact_dir))

    artifact_dir = TMPDIR or tempfile.mkdtemp()
    logger.info(
        f"The `artifact_dir` was not provided and "
        f"automatically set to: {artifact_dir}"
    )

    return artifact_dir


class GenericModel(MetadataMixin, Introspectable, EvaluatorMixin):
    """Generic Model class which is the base class for all the frameworks including
    the unsupported frameworks.

    Attributes
    ----------
    algorithm: str
        The algorithm of the model.
    artifact_dir: str
        Artifact directory to store the files needed for deployment.
    auth: Dict
        Default authentication is set using the `ads.set_auth` API. To override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create
        an authentication signer to instantiate an IdentityClient object.
    estimator: Callable
        Any model object generated by sklearn framework
    framework: str
        The framework of the model.
    hyperparameter: dict
        The hyperparameters of the estimator.
    metadata_custom: ModelCustomMetadata
        The model custom metadata.
    metadata_provenance: ModelProvenanceMetadata
        The model provenance metadata.
    metadata_taxonomy: ModelTaxonomyMetadata
        The model taxonomy metadata.
    model_artifact: ModelArtifact
        This is built by calling prepare.
    model_deployment: ModelDeployment
        A ModelDeployment instance.
    model_file_name: str
        Name of the serialized model.
    model_id: str
        The model ID.
    model_input_serializer: SERDE
        Instance of ads.model.SERDE. Used for serialize/deserialize data.
    properties: ModelProperties
        ModelProperties object required to save and deploy model.
    runtime_info: RuntimeInfo
        A RuntimeInfo instance.
    schema_input: Schema
        Schema describes the structure of the input data.
    schema_output: Schema
        Schema describes the structure of the output data.
    serialize: bool
        Whether to serialize the model to pkl file by default. If False, you need to serialize the model manually,
        save it under artifact_dir and update the score.py manually.
    version: str
        The framework version of the model.

    Methods
    -------
    delete_deployment(...)
        Deletes the current model deployment.
    deploy(..., **kwargs)
        Deploys a model.
    from_model_artifact(uri, ..., **kwargs)
        Loads model from the specified folder, or zip/tar archive.
    from_model_catalog(model_id, ..., **kwargs)
        Loads model from model catalog.
    from_model_deployment(model_deployment_id, ..., **kwargs)
        Loads model from model deployment.
    update_deployment(model_deployment_id, ..., **kwargs)
        Updates a model deployment.
    from_id(ocid, ..., **kwargs)
        Loads model from model OCID or model deployment OCID.
    introspect(...)
        Runs model introspection.
    predict(data, ...)
        Returns prediction of input data run against the model deployment endpoint.
    prepare(..., **kwargs)
        Prepare and save the score.py, serialized model and runtime.yaml file.
    prepare_save_deploy(..., **kwargs)
        Shortcut for prepare, save and deploy steps.
    reload(...)
        Reloads the model artifact files: `score.py` and the `runtime.yaml`.
    restart_deployment(...)
        Restarts the model deployment.
    save(..., **kwargs)
        Saves model artifacts to the model catalog.
    set_model_input_serializer(serde)
        Registers serializer used for serializing data passed in verify/predict.
    summary_status(...)
        Gets a summary table of the current status.
    verify(data, ...)
        Tests if deployment works in local environment.
    upload_artifact(...)
        Uploads model artifacts to the provided `uri`.
    download_artifact(...)
        Downloads model artifacts from the model catalog.
    update_summary_status(...)
        Update the status in the summary table.
    update_summary_action(...)
        Update the actions needed from the user in the summary table.


    Examples
    --------
    >>> import tempfile
    >>> from ads.model.generic_model import GenericModel

    >>> class Toy:
    ...     def predict(self, x):
    ...         return x ** 2
    >>> estimator = Toy()

    >>> model = GenericModel(estimator=estimator, artifact_dir=tempfile.mkdtemp())
    >>> model.summary_status()
    >>> model.prepare(
    ...     inference_conda_env="dbexp_p38_cpu_v1",
    ...     inference_python_version="3.8",
    ...     model_file_name="toy_model.pkl",
    ...     training_id=None,
    ...     force_overwrite=True
    ... )
    >>> model.verify(2)
    >>> model.save()
    >>> model.deploy()
    >>> # Update access log id, freeform tags and description for the model deployment
    >>> model.update_deployment(
    ...     access_log={
    ...         log_id=<log_ocid>
    ...     },
    ...     description="Description for Custom Model",
    ...     freeform_tags={"key": "value"},
    ... )
    >>> model.predict(2)
    >>> # Uncomment the line below to delete the model and the associated model deployment
    >>> # model.delete(delete_associated_model_deployment = True)
    """

    _summary_status = None
    _PREFIX = "generic"
    model_input_serializer_type = ModelInputSerializerType
    model_save_serializer_type = ModelSerializerType

    def __init__(
        self,
        estimator: Callable = None,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Optional[Dict] = None,
        serialize: bool = True,
        model_save_serializer: Optional[SERDE] = None,
        model_input_serializer: Optional[SERDE] = None,
        **kwargs: dict,
    ) -> Self:
        """GenericModel Constructor.

        Parameters
        ----------
        estimator: (Callable).
            Trained model.
        artifact_dir: (str, optional). Defaults to None.
            Artifact directory to store the files needed for deployment.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        serialize: (bool, optional). Defaults to True.
            Whether to serialize the model to pkl file by default. If False, you need to serialize the model manually,
            save it under artifact_dir and update the score.py manually.
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model input.
        """
        if (
            artifact_dir
            and ObjectStorageDetails.is_oci_path(artifact_dir)
            and not self._PREFIX == "spark"
        ):
            raise ValueError(
                f"Unsupported value of `artifact_dir`: {artifact_dir}. "
                "Only SparkPipelineModel framework supports object storage path as `artifact_dir`."
            )

        self.estimator = estimator
        self.auth = auth or authutil.default_signer()
        self.dsc_model = (
            DataScienceModel()
            .with_custom_metadata_list(ModelCustomMetadata())
            .with_provenance_metadata(ModelProvenanceMetadata())
            .with_defined_metadata_list(ModelTaxonomyMetadata())
            .with_input_schema(Schema())
            .with_output_schema(Schema())
        )

        self.model_file_name = None
        self.artifact_dir = (
            artifact_dir
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else _prepare_artifact_dir(artifact_dir)
        )
        self.local_copy_dir = (
            _prepare_artifact_dir()
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else self.artifact_dir
        )
        if ObjectStorageDetails.is_oci_path(self.artifact_dir):
            os.environ["OCI_DEPLOYMENT_PATH"] = self.artifact_dir

        self.model_artifact = None
        self.framework = None
        self.algorithm = None
        self.version = None
        self.hyperparameter = None
        self._introspect = ModelIntrospect(self)
        self.model_deployment = (
            ModelDeployment()
            .with_infrastructure(ModelDeploymentInfrastructure())
            .with_runtime(ModelDeploymentContainerRuntime())
        )
        self.runtime_info = None
        self._as_onnx = kwargs.pop("as_onnx", False)
        self._score_args = {}

        if properties:
            self.properties = (
                properties
                if isinstance(properties, ModelProperties)
                else ModelProperties.from_dict(properties)
            )
        else:
            self.properties = ModelProperties().with_env()

        self._serialize = serialize
        self._summary_status = SummaryStatus()
        self._init_serde(
            model_input_serde=model_input_serializer,
            model_save_serializer=model_save_serializer,
        )
        self.ignore_conda_error = False

    def _init_serde(
        self,
        model_input_serde: Union[SERDE, str] = None,
        model_save_serializer: Union[SERDE, str] = None,
    ):
        """Initializes serde.

        Parameters
        ----------
        model_save_serializer: (SERDE or str). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE or str). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model input.
        """
        if model_input_serde is None:
            logger.warning(
                "In the future model input will be serialized by `cloudpickle` by "
                "default. Currently, model input are serialized into a dictionary "
                "containing serialized input data and original data type information."
                'Set `model_input_serializer="cloudpickle"` to use cloudpickle model input serializer.'
            )
        self.set_model_input_serializer(
            model_input_serializer=model_input_serde
            or self.model_input_serializer_type.JSON
        )
        self.set_model_save_serializer(
            model_save_serializer or self.model_save_serializer_type.CLOUDPICKLE
        )

    @property
    def metadata_custom(self):
        return self.dsc_model.custom_metadata_list

    @metadata_custom.setter
    def metadata_custom(self, value: ModelCustomMetadata):
        self.dsc_model.with_custom_metadata_list(value)

    @property
    def metadata_taxonomy(self):
        return self.dsc_model.defined_metadata_list

    @metadata_taxonomy.setter
    def metadata_taxonomy(self, value: ModelTaxonomyMetadata):
        self.dsc_model.with_defined_metadata_list(value)

    @property
    def metadata_provenance(self):
        return self.dsc_model.provenance_metadata

    @metadata_provenance.setter
    def metadata_provenance(self, value: ModelProvenanceMetadata):
        self.dsc_model.with_provenance_metadata(value)

    @property
    def schema_input(self):
        return self.dsc_model.input_schema

    @schema_input.setter
    def schema_input(self, value: Schema):
        self.dsc_model.with_input_schema(value)

    @property
    def schema_output(self):
        return self.dsc_model.output_schema

    @schema_output.setter
    def schema_output(self, value: Schema):
        self.dsc_model.with_output_schema(value)

    @property
    def model_id(self):
        return self.dsc_model.id

    @property
    def model_deployment_id(self):
        if self.model_deployment:
            return self.model_deployment.model_deployment_id
        return None

    def __repr__(self) -> str:
        """Representation of the model."""
        return self._to_yaml()

    def _to_dict(self):
        """Converts the model attributes to dictionary format."""
        attributes = {}
        for key in _ATTRIBUTES_TO_SHOW_:
            if key == "artifact_dir":
                attributes[key] = {getattr(self, key): [self._get_files()]}
            else:
                attributes[key] = getattr(self, key, None)
        return attributes

    def _to_yaml(self):
        """Converts the model attributes to yaml format."""
        return yaml.safe_dump(self._to_dict())

    def set_model_input_serializer(
        self,
        model_input_serializer: Union[str, SERDE],
    ):
        """Registers serializer used for serializing data passed in verify/predict.

        Examples
        --------
        >>> generic_model.set_model_input_serializer(GenericModel.model_input_serializer_type.CLOUDPICKLE)

        >>> # Register serializer by passing the name of it.
        >>> generic_model.set_model_input_serializer("cloudpickle")

        >>> # Example of creating customized model input serializer and registering it.
        >>> from ads.model import SERDE
        >>> from ads.model.generic_model import GenericModel

        >>> class MySERDE(SERDE):
        ...     def __init__(self):
        ...         super().__init__()
        ...     def serialize(self, data):
        ...         serialized_data = 1
        ...         return serialized_data
        ...     def deserialize(self, data):
        ...         deserialized_data = 2
        ...         return deserialized_data

        >>> class Toy:
        ...     def predict(self, x):
        ...         return x ** 2

        >>> generic_model = GenericModel(
        ...    estimator=Toy(),
        ...    artifact_dir=tempfile.mkdtemp(),
        ...    model_input_serializer=MySERDE()
        ... )

        >>> # Or register the serializer after creating model instance.
        >>> generic_model.set_model_input_serializer(MySERDE())

        Parameters
        ----------
        model_input_serializer: (str, or ads.model.SERDE)
            name of the serializer, or instance of SERDE.
        """
        if isinstance(model_input_serializer, str):
            self.model_input_serializer = ModelInputSerializerFactory.get(
                model_input_serializer
            )
        else:
            self.model_input_serializer = model_input_serializer

        try:
            serializer_name = self.model_input_serializer.name
            if serializer_name not in SUPPORTED_MODEL_INPUT_SERIALIZERS:
                logger.warn(
                    "Replace the code of `deserialize()` in `score.py` with "
                    "the your own implementation of `deserialize()`."
                )
        except AttributeError:
            self.model_input_serializer.name = "customized"
            logger.warn(
                "Model input will be serialized by `serialize()` "
                "defined in your provided `model_input_serializer`. "
                "Replace the code of `deserialize()` in `score.py` with "
                "the your own implementation of `deserialize()`."
            )

    def set_model_save_serializer(self, model_save_serializer: Union[str, SERDE]):
        """Registers serializer used for saving model.

        Examples
        --------
        >>> generic_model.set_model_save_serializer(GenericModel.model_save_serializer_type.CLOUDPICKLE)

        >>> # Register serializer by passing the name of it.
        >>> generic_model.set_model_save_serializer("cloudpickle")

        >>> # Example of creating customized model save serializer and registing it.
        >>> from ads.model import SERDE
        >>> from ads.model.generic_model import GenericModel

        >>> class MySERDE(SERDE):
        ...     def __init__(self):
        ...         super().__init__()
        ...     def serialize(self, data):
        ...         serialized_data = 1
        ...         return serialized_data
        ...     def deserialize(self, data):
        ...         deserialized_data = 2
        ...         return deserialized_data

        >>> class Toy:
        ...     def predict(self, x):
        ...         return x ** 2

        >>> generic_model = GenericModel(
        ...    estimator=Toy(),
        ...    artifact_dir=tempfile.mkdtemp(),
        ...    model_save_serializer=MySERDE()
        ... )

        >>> # Or register the serializer after creating model instance.
        >>> generic_model.set_model_save_serializer(MySERDE())

        Parameters
        ----------
        model_save_serializer: (ads.model.SERDE or str)
            name of the serializer or instance of SERDE.
        """
        if isinstance(model_save_serializer, str):
            self.model_save_serializer = ModelSerializerFactory.get(
                model_save_serializer
            )
        else:
            self.model_save_serializer = model_save_serializer

        try:
            serializer_name = self.model_save_serializer.name
            if serializer_name not in SUPPORTED_MODEL_SERIALIZERS:
                logger.warn(
                    "Replace the code of `load_model()` in `score.py` with "
                    "the your own implementation of `deserialize()`."
                )
        except AttributeError:
            self.model_save_serializer.name = "customized"
            logger.warn(
                "Model will be saved by `serialize()` "
                "defined in your provided `model_save_serializer`. "
                "Replace the code of `load_model()` in `score.py` with "
                "the your own implementation of `deserialize()`."
            )

    def serialize_model(
        self,
        as_onnx: bool = False,
        initial_types: List[Tuple] = None,
        force_overwrite: bool = False,
        X_sample: any = None,
        **kwargs,
    ):
        """
        Serialize and save model using ONNX or model specific method.

        Parameters
        ----------
        as_onnx: (boolean, optional)
            If set as True, convert into ONNX model.
        initial_types: (List[Tuple], optional)
            a python list. Each element is a tuple of a variable name and a data type.
        force_overwrite: (boolean, optional)
            If set as True, overwrite serialized model if exists.
        X_sample: (any, optional). Defaults to None.
            Contains model inputs such that model(X_sample) is a valid
            invocation of the model, used to valid model input type.

        Returns
        -------
        None
            Nothing
        """
        if self._serialize:
            if not self.model_file_name:
                self.model_file_name = self._handle_model_file_name(as_onnx=as_onnx)
            if not self.estimator:
                raise ValueError(
                    "Parameter `estimator` has to be provided when `serialize=True`, or you can set `serialize=False`."
                )
            self._serialize_model_helper(
                initial_types, force_overwrite, X_sample, **kwargs
            )
        else:
            raise SerializeModelNotImplementedError(
                "`serialize_model` is not implemented."
            )

    def _serialize_model_helper(
        self,
        initial_types: List[Tuple] = None,
        force_overwrite: bool = False,
        X_sample: any = None,
        **kwargs,
    ):
        model_path = self._check_model_file(
            self.model_file_name, force_overwrite=force_overwrite
        )
        self.get_model_serializer().serialize(
            estimator=self.estimator,
            model_path=model_path,
            X_sample=X_sample,
            initial_types=initial_types,
            **kwargs,
        )

    def _check_model_file(self, model_file_name, force_overwrite):
        model_path = os.path.join(self.artifact_dir, model_file_name)
        if utils.is_path_exists(uri=model_path, auth=self.auth) and not force_overwrite:
            raise ValueError(
                f"The {model_path} already exists, set force_overwrite to True if you wish to overwrite."
            )
        if not ObjectStorageDetails.is_oci_path(self.artifact_dir):
            os.makedirs(self.artifact_dir, exist_ok=True)
        return model_path

    def _handle_model_file_name(self, as_onnx: bool, model_file_name: str = None):
        if as_onnx:
            self._set_model_save_serializer_to_onnx()

        if not model_file_name:
            if not self._serialize:
                raise NotImplementedError("`model_file_name` has to be provided.")
            else:
                model_file_name = f"model.{self._get_model_file_suffix()}"

        if as_onnx:
            assert model_file_name.endswith(
                ".onnx"
            ), "Wrong file extension. Expecting `.onnx` suffix."

        return model_file_name

    def _get_model_file_suffix(self):
        try:
            suffix = self.model_save_serializer.model_file_suffix
            return suffix
        except AttributeError as e:
            logger.error(
                "Please specify `model_file_suffix` in `model_save_serializer`. "
            )
            raise e

    def _set_model_save_serializer_to_onnx(self):
        try:
            self.set_model_save_serializer(self.model_save_serializer_type.ONNX)
        except AttributeError as e:
            logger.error(
                f"This framework {self._PREFIX} to Onnx Conversion is not supported. Please set `as_onnx=False` (default) to perform other model serialization."
            )
            raise e

    def _onnx_data_transformer(
        self,
        X: Union[pd.DataFrame, pd.Series],
        impute_values: Dict = None,
        force_overwrite: bool = False,
    ):
        """Apply onnx data transformer to data."""
        if self.framework in FRAMEWORKS_WITHOUT_ONNX_DATA_TRANSFORM or X is None:
            return X
        try:
            if hasattr(self, "onnx_data_preprocessor") and isinstance(
                self.onnx_data_preprocessor, ONNXTransformer
            ):
                X = self.onnx_data_preprocessor.transform(X=X)

            self.onnx_data_preprocessor = ONNXTransformer()
            X = self.onnx_data_preprocessor.fit_transform(
                X=X, impute_values=impute_values
            )
            if (
                os.path.exists(os.path.join(self.artifact_dir, ONNX_DATA_TRANSFORMER))
                and not force_overwrite
            ):
                raise ValueError(
                    f"{ONNX_DATA_TRANSFORMER} already exists. "
                    "Set `force_overwrite` to True if you wish to overwrite."
                )
            else:
                try:
                    self.onnx_data_preprocessor.save(
                        os.path.join(self.artifact_dir, ONNX_DATA_TRANSFORMER)
                    )
                except Exception as e:
                    logger.error(
                        f"Unable to serialize the data transformer due to: {e}."
                    )
                    raise e
        except Exception as e:
            logger.warn(f"Onnx Data Transformation was unsuccessful with error: {e}")
            raise e
        return X

    def prepare(
        self,
        inference_conda_env: str = None,
        inference_python_version: str = None,
        training_conda_env: str = None,
        training_python_version: str = None,
        model_file_name: str = None,
        as_onnx: bool = False,
        initial_types: List[Tuple] = None,
        force_overwrite: bool = False,
        namespace: str = CONDA_BUCKET_NS,
        use_case_type: str = None,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        training_script_path: str = None,
        training_id: str = _TRAINING_RESOURCE_ID,
        ignore_pending_changes: bool = True,
        max_col_num: int = DATA_SCHEMA_MAX_COL_NUM,
        ignore_conda_error: bool = False,
        score_py_uri: str = None,
        **kwargs: Dict,
    ) -> "GenericModel":
        """Prepare and save the score.py, serialized model and runtime.yaml file.

        Parameters
        ----------
        inference_conda_env: (str, optional). Defaults to None.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
        inference_python_version: (str, optional). Defaults to None.
            Python version which will be used in deployment.
        training_conda_env: (str, optional). Defaults to None.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
            If `training_conda_env` is not provided, `training_conda_env` will
            use the same value of `training_conda_env`.
        training_python_version: (str, optional). Defaults to None.
            Python version used during training.
        model_file_name: (str, optional). Defaults to `None`.
            Name of the serialized model.
            Will be auto generated if not provided.
        as_onnx: (bool, optional). Defaults to False.
            Whether to serialize as onnx model.
        initial_types: (list[Tuple], optional).
            Defaults to None. Only used for SklearnModel, LightGBMModel and XGBoostModel.
            Each element is a tuple of a variable name and a type.
            Check this link http://onnx.ai/sklearn-onnx/api_summary.html#id2 for
            more explanation and examples for `initial_types`.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files.
        namespace: (str, optional).
            Namespace of region. This is used for identifying which region the service pack
            is from when you pass a slug to inference_conda_env and training_conda_env.
        use_case_type: str
            The use case type of the model. Use it through UserCaseType class or string provided in `UseCaseType`. For
            example, use_case_type=UseCaseType.BINARY_CLASSIFICATION or use_case_type="binary_classification". Check
            with UseCaseType class to see all supported types.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema.
        y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of output data that will be used to generate output schema.
        training_script_path: str. Defaults to None.
            Training script path.
        training_id: (str, optional). Defaults to value from environment variables.
            The training OCID for model. Can be notebook session or job OCID.
        ignore_pending_changes: bool. Defaults to False.
            whether to ignore the pending changes in the git.
        max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
            Do not generate the input schema if the input has more than this
            number of features(columns).
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        score_py_uri: (str, optional). Defaults to None.
            The uri of the customized score.py, which can be local path or OCI object storage URI.
            When provide with this attibute, the `score.py` will not be auto generated, and the
            provided `score.py` will be added into artifact_dir.
        kwargs:
            impute_values: (dict, optional).
                The dictionary where the key is the column index(or names is accepted
                for pandas dataframe) and the value is the impute value for the corresponding column.

        Raises
        ------
        FileExistsError
            If files already exist but `force_overwrite` is False.
        ValueError
            If `inference_python_version` is not provided, but also cannot be found
            through manifest file.

        Returns
        -------
        GenericModel
            An instance of `GenericModel` class.
        """
        # Populate properties from args and kwargs.
        # empty values will be ignored.

        locals_dict = _extract_locals(locals())
        locals_dict.pop("training_id", None)
        self.properties.with_dict(locals_dict)

        if training_id != _TRAINING_RESOURCE_ID:
            self.properties.training_id = training_id
        elif not self.properties.training_id:
            self.properties.training_id = _TRAINING_RESOURCE_ID

        self.ignore_conda_error = ignore_conda_error
        if self.ignore_conda_error:
            logger.info(
                "`ignore_conda_error` is set to True and `.verify()` is targeted to test the generated score.py on the local conda environment, not the container."
            )
        if not self.properties.inference_conda_env:
            try:
                conda_prefix = os.environ.get("CONDA_PREFIX", None)
                manifest = fetch_manifest_from_conda_location(conda_prefix)
                if "pack_path" in manifest:
                    self.properties.inference_conda_env = manifest["pack_path"]
                else:
                    if not self.ignore_conda_error:
                        raise ValueError(
                            "`inference_conda_env` must be specified for conda runtime. If you are using container runtime, set `ignore_conda_error=True`."
                        )
                self.properties.inference_python_version = (
                    manifest["python"]
                    if "python" in manifest
                    and not self.properties.inference_python_version
                    else self.properties.inference_python_version
                )
            except:
                if not self.ignore_conda_error:
                    raise ValueError(
                        "`inference_conda_env` must be specified for conda runtime. If you are using container runtime, set `ignore_conda_error=True`."
                    )

        self._as_onnx = as_onnx
        if as_onnx:
            self._set_model_save_serializer_to_onnx()

        self.model_file_name = self._handle_model_file_name(
            as_onnx=as_onnx, model_file_name=model_file_name
        )
        if (
            not isinstance(self.model_file_name, str)
            or self.model_file_name.strip() == ""
        ):
            raise ValueError("The `model_file_name` needs to be provided.")

        if not ObjectStorageDetails.is_oci_path(self.artifact_dir):
            os.makedirs(self.artifact_dir, exist_ok=True)

        # Bring in .model-ignore file
        uri_src = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "common/.model-ignore",
        )
        uri_dst = os.path.join(self.artifact_dir, ".model-ignore")
        utils.copy_file(uri_src=uri_src, uri_dst=uri_dst, force_overwrite=True)

        self.model_artifact = ModelArtifact(
            artifact_dir=self.artifact_dir,
            model_file_name=self.model_file_name,
            auth=self.auth,
            local_copy_dir=self.local_copy_dir,
        )
        try:
            self.runtime_info = self.model_artifact.prepare_runtime_yaml(
                inference_conda_env=self.properties.inference_conda_env,
                inference_python_version=self.properties.inference_python_version,
                training_conda_env=self.properties.training_conda_env,
                training_python_version=self.properties.training_python_version,
                force_overwrite=force_overwrite,
                namespace=namespace,
                bucketname=DEFAULT_CONDA_BUCKET_NAME,
                auth=self.auth,
                ignore_conda_error=self.ignore_conda_error,
            )
        except ValueError as e:
            raise e

        self.update_summary_status(
            detail=PREPARE_STATUS_GEN_RUNTIME_DETAIL, status=ModelState.DONE.value
        )

        if self.estimator:
            if as_onnx:
                X_sample = self._onnx_data_transformer(
                    X_sample,
                    impute_values=kwargs.pop("impute_values", {}),
                    force_overwrite=force_overwrite,
                )
            try:
                self.serialize_model(
                    as_onnx=as_onnx,
                    force_overwrite=force_overwrite,
                    initial_types=initial_types,
                    X_sample=X_sample,
                    **kwargs,
                )
                self.update_summary_status(
                    detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                    status=ModelState.DONE.value,
                )
            except SerializeModelNotImplementedError as e:
                if not utils.is_path_exists(
                    uri=os.path.join(self.artifact_dir, self.model_file_name),
                    auth=self.auth,
                ):
                    self.update_summary_action(
                        detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                        action=(
                            "Model is not automatically serialized. "
                            f"Serialize the model as `{self.model_file_name}` and "
                            f"save to the {self.artifact_dir}."
                        ),
                    )
                    self.update_summary_status(
                        detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                        status=ModelState.NEEDSACTION.value,
                    )
                    logger.warning(
                        f"{self.model_file_name} not found in {self.artifact_dir}. "
                        f"Save the serialized model under {self.artifact_dir}."
                    )
                    self.update_summary_action(
                        detail=PREPARE_STATUS_GEN_SCORE_DETAIL,
                        action=(
                            "`load_model` is not automatically generated. "
                            "Finish implementing it and call .verify to check if it works."
                        ),
                    )
            except Exception as e:
                raise e

        if as_onnx:
            jinja_template_filename = "score_onnx_new"
        else:
            if self.framework and self.framework != "other":
                jinja_template_filename = "score_" + self.framework
                if self.framework == "transformers":
                    jinja_template_filename = "score_" + "huggingface_pipeline"
            else:
                jinja_template_filename = (
                    "score-pkl" if self._serialize else "score_generic"
                )

        if score_py_uri:
            utils.copy_file(
                uri_src=score_py_uri,
                uri_dst=os.path.join(self.artifact_dir, "score.py"),
                force_overwrite=force_overwrite,
                auth=self.auth,
            )
        else:
            self.model_artifact.prepare_score_py(
                jinja_template_filename=jinja_template_filename,
                model_file_name=self.model_file_name,
                data_deserializer=self.model_input_serializer.name,
                model_serializer=self.model_save_serializer.name,
                auth=self.auth,
                **{**kwargs, **self._score_args},
            )

        self.update_summary_status(
            detail=PREPARE_STATUS_GEN_SCORE_DETAIL, status=ModelState.DONE.value
        )

        self.populate_metadata(
            use_case_type=use_case_type,
            X_sample=X_sample,
            y_sample=y_sample,
            training_script_path=self.properties.training_script_path,
            training_id=self.properties.training_id,
            ignore_pending_changes=ignore_pending_changes,
            max_col_num=max_col_num,
            ignore_conda_error=self.ignore_conda_error,
            auth=self.auth,
        )

        self.update_summary_status(
            detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
            status=ModelState.DONE.value,
        )

        self.update_summary_status(
            detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
            status=ModelState.AVAILABLE.value,
        )

        if not self.ignore_conda_error:
            self.update_summary_status(
                detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
                status=ModelState.AVAILABLE.value,
            )

        self.update_summary_status(
            detail=SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL,
            status=ModelState.AVAILABLE.value,
        )
        return self

    def _handle_input_data(
        self, data: Any = None, auto_serialize_data: bool = True, **kwargs
    ):
        """Handle input data and serialize it as required.

        Parameters
        ----------
        data: Any
            Data for the prediction.
        auto_serialize_data: bool
            Defaults to True. Indicate whether to serialize the input data.

        kwargs:
            storage_options: dict
                Passed to ADSImage.open.

        Raises
        ------
        TypeError:
            `data` is not json serializable or bytes. Set `auto_serialize_data` to `True` to serialize the input data.
        ValueError:
            Either use `image` argument through kwargs to pass in image file or use `data` argument to pass the data.

        Returns
        -------
        object: Data used for a request.
        """
        if isinstance(data, bytes):
            return data
        if not auto_serialize_data:
            if not _is_json_serializable(data) and not isinstance(data, bytes):
                raise TypeError(
                    "`data` is not json serializable or bytes. Set `auto_serialize_data` to `True` to serialize the input data."
                )
            return data

        if data is None and "image" not in kwargs.keys():
            raise ValueError(
                "Either use `image` argument through kwargs to pass in image file or use `data` argument to pass the data."
            )

        if "image" in kwargs.keys():
            data = self._handle_image_input(image=kwargs.pop("image"), **kwargs)

        serialized_data = self.model_input_serializer.serialize(data=data, **kwargs)
        return serialized_data

    def _handle_image_input(self, image, **kwargs):
        """Validates the image input and converts it to tensor.

        Parameters
        ----------
        image: PIL.Image Object or uri.
            image file path or opened image file.

        kwargs:
            storage_options: dict
                Passed to ADSImage.open.

        Raises
        ------
        ValueError: Cannot open or identify the given image file.

        Returns
        -------
        tensor: tf.tensor or torch.tensor.
        """
        if not isinstance(image, Image.Image):
            try:
                image = ADSImage.open(
                    path=image, storage_options=kwargs.pop("storage_options", {})
                ).img
            except Exception as e:
                raise ValueError(
                    f"Cannot open or identify the given image file. See details: {e}"
                )
        tensor = self._to_tensor(image)
        return tensor

    def _to_tensor(self, data):
        """Only PyTorchModel and TensorflowModel will implement this method.

        Args:
            data (Any): Data needs to be converted to tensor.

        Raises:
            NotImplementedError: Only PyTorchModel and TensorflowModel will implement this method.
        """
        raise NotImplementedError(
            "Only PyTorchModel and TensorflowModel will implement this method."
        )

    def get_data_serializer(self):
        """Gets data serializer.

        Returns
        -------
        object: ads.model.Serializer object.
        """
        return self.model_input_serializer

    def get_model_serializer(self):
        """Gets model serializer."""
        return self.model_save_serializer

    def verify(
        self,
        data: Any = None,
        reload_artifacts: bool = True,
        auto_serialize_data: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Test if deployment works in local environment.

        Examples
        --------
        >>> uri = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        >>> prediction = model.verify(image=uri)['prediction']

        >>> # examples on storage options
        >>> prediction = model.verify(
        ...        image="oci://<bucket>@<tenancy>/myimage.png",
        ...        storage_options=ads.auth.default_signer()
        ... )['prediction']

        Parameters
        ----------
        data: Any
            Data used to test if deployment works in local environment.
        reload_artifacts: bool. Defaults to True.
            Whether to reload artifacts or not.
        is_json_payload: bool
            Defaults to False. Indicate whether to send data with a `application/json` MIME TYPE.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Defauls to `False` for GenericModel, and `True` for other frameworks.
            `data` required to be json serializable if `auto_serialize_data=False`.
            if `auto_serialize_data` set to True, data will be serialized before sending to model deployment endpoint.
        kwargs:
            content_type: str, used to indicate the media type of the resource.
            image: PIL.Image Object or uri for the image.
               A valid string path for image file can be local path, http(s), oci, s3, gs.
            storage_options: dict
               Passed to `fsspec.open` for a particular storage connection.
               Please see `fsspec` (https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open) for more details.

        Returns
        -------
        Dict
            A dictionary which contains prediction results.
        """
        if self.model_artifact is None:
            raise ArtifactsNotAvailableError

        endpoint = f"http://127.0.0.1:8000/predict"
        data = self._handle_input_data(data, auto_serialize_data, **kwargs)

        request_body = send_request(
            data,
            endpoint,
            dry_run=True,
            is_json_payload=_is_json_serializable(data),
            **kwargs,
        )

        if reload_artifacts:
            self.model_artifact.reload()

        prediction = self.model_artifact.predict(request_body)

        try:
            requests.Request("POST", endpoint, json=prediction)
        except:
            raise TypeError(
                "The prediction result is not json serializable. "
                "Please modify the score.py."
            )

        self.update_summary_status(
            detail=VERIFY_STATUS_LOCAL_TEST_DETAIL, status=ModelState.DONE.value
        )
        return prediction

    def introspect(self) -> pd.DataFrame:
        """Conducts instrospection.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame which contains the instrospection results.
        """
        df = self._introspect()
        return df

    @classmethod
    def from_model_artifact(
        cls: Type[Self],
        uri: str,
        model_file_name: str = None,
        artifact_dir: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        properties: Optional[ModelProperties] = None,
        ignore_conda_error: Optional[bool] = False,
        **kwargs: dict,
    ) -> Self:
        """Loads model from a folder, or zip/tar archive.

        Parameters
        ----------
        uri: str
            The folder path, ZIP file path, or TAR file path. It could contain a
            seriliazed model(required) as well as any files needed for deployment including:
            serialized model, runtime.yaml, score.py and etc. The content of the folder will be
            copied to the `artifact_dir` folder.
        model_file_name: (str, optional). Defaults to `None`.
            The serialized model file name.
            Will be extracted from artifacts if not provided.
        artifact_dir: (str, optional). Defaults to `None`.
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.

        Returns
        -------
        Self
            An instance of `GenericModel` class.

        Raises
        ------
        ValueError
            If `model_file_name` not provided.
        """
        if (
            cls._PREFIX != "spark"
            and artifact_dir
            and ObjectStorageDetails.is_oci_path(artifact_dir)
        ):
            raise ValueError(
                f"Unsupported value of `artifact_dir`: {artifact_dir}. "
                "Only SparkPipelineModel framework supports object storage path as artifact_dir."
            )

        local_vars = _extract_locals(locals())
        properties = properties or ModelProperties()
        properties.with_dict(local_vars)
        auth = auth or authutil.default_signer()
        artifact_dir = _prepare_artifact_dir(artifact_dir)
        reload = kwargs.pop("reload", False)
        model_artifact = ModelArtifact.from_uri(
            uri=uri,
            artifact_dir=artifact_dir,
            auth=auth,
            force_overwrite=force_overwrite,
            ignore_conda_error=ignore_conda_error,
            model_file_name=model_file_name,
            reload=reload,
        )
        model = cls(
            estimator=model_artifact.model,
            artifact_dir=artifact_dir,
            auth=auth,
            properties=properties,
            **kwargs,
        )
        model.model_file_name = model_file_name or model_artifact.model_file_name
        model.local_copy_dir = model_artifact.local_copy_dir
        model.model_artifact = model_artifact
        model.ignore_conda_error = ignore_conda_error

        if reload:
            model.reload_runtime_info()
            model.update_summary_action(
                detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
                action="Call .populate_metadata() to populate metadata.",
            )

        model.update_summary_status(
            detail=PREPARE_STATUS_GEN_SCORE_DETAIL,
            status=ModelState.NOTAPPLICABLE.value,
        )
        model.update_summary_status(
            detail=PREPARE_STATUS_GEN_RUNTIME_DETAIL,
            status=ModelState.NOTAPPLICABLE.value,
        )
        model.update_summary_status(
            detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
            status=ModelState.NOTAPPLICABLE.value,
        )
        model.update_summary_status(
            detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
            status=ModelState.AVAILABLE.value
            if reload
            else ModelState.NOTAPPLICABLE.value,
        )

        return model

    def download_artifact(
        self,
        artifact_dir: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        **kwargs,
    ) -> "GenericModel":
        """Downloads model artifacts from the model catalog.

        Parameters
        ----------
        artifact_dir: (str, optional). Defaults to `None`.
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.

        Returns
        -------
        Self
            An instance of `GenericModel` class.

        Raises
        ------
        ValueError
            If `model_id` is not available in the GenericModel object.
        """
        model_id = self.model_id
        if not model_id:
            raise ValueError(
                "`model_id` is not available, load the GenericModel object first."
            )

        if not artifact_dir:
            artifact_dir = self.artifact_dir
        artifact_dir = _prepare_artifact_dir(artifact_dir)

        target_dir = (
            _prepare_artifact_dir()
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else artifact_dir
        )

        dsc_model = DataScienceModel.from_id(model_id)
        dsc_model.download_artifact(
            target_dir=target_dir,
            force_overwrite=force_overwrite,
            bucket_uri=bucket_uri,
            remove_existing_artifact=remove_existing_artifact,
            auth=auth,
            region=kwargs.pop("region", None),
            timeout=kwargs.pop("timeout", None),
        )
        model_artifact = ModelArtifact.from_uri(
            uri=target_dir,
            artifact_dir=artifact_dir,
            model_file_name=self.model_file_name,
            force_overwrite=force_overwrite,
            auth=auth,
            ignore_conda_error=self.ignore_conda_error,
        )
        self.dsc_model = dsc_model
        self.local_copy_dir = model_artifact.local_copy_dir
        self.model_artifact = model_artifact
        self.reload_runtime_info()

        self.update_summary_status(
            detail=PREPARE_STATUS_GEN_SCORE_DETAIL,
            status=ModelState.DONE.value,
        )
        self.update_summary_status(
            detail=PREPARE_STATUS_GEN_RUNTIME_DETAIL,
            status=ModelState.DONE.value,
        )
        self.update_summary_status(
            detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL, status=ModelState.DONE.value
        )
        self.update_summary_status(
            detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
            status=ModelState.DONE.value,
        )
        self.update_summary_status(
            detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
            status=ModelState.AVAILABLE.value,
        )
        self.update_summary_action(
            detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
            action="",
        )
        self.update_summary_status(
            detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
            status=ModelState.AVAILABLE.value,
        )
        self.update_summary_status(
            detail=SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL,
            status=ModelState.AVAILABLE.value,
        )
        return self

    @classmethod
    def from_model_catalog(
        cls: Type[Self],
        model_id: str,
        model_file_name: str = None,
        artifact_dir: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        properties: Optional[Union[ModelProperties, Dict]] = None,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        ignore_conda_error: Optional[bool] = False,
        download_artifact: Optional[bool] = True,
        **kwargs,
    ) -> Self:
        """Loads model from model catalog.

        Parameters
        ----------
        model_id: str
            The model OCID.
        model_file_name: (str, optional). Defaults to `None`.
            The name of the serialized model.
        artifact_dir: (str, optional). Defaults to `None`.
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        download_artifact: (bool, optional). Defaults to True.
            Whether to download the model pickle or checkpoints
        kwargs:
            compartment_id : (str, optional)
                Compartment OCID. If not specified, the value will be taken from the environment variables.
            timeout : (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.
            region: (str, optional). Defaults to `None`.
                The destination Object Storage bucket region.
                By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.

        Returns
        -------
        Self
            An instance of GenericModel class.
        """
        if (
            cls._PREFIX != "spark"
            and artifact_dir
            and ObjectStorageDetails.is_oci_path(artifact_dir)
        ):
            raise ValueError(
                f"Unsupported value of `artifact_dir`: {artifact_dir}. "
                "Only SparkPipelineModel framework supports object storage path as artifact_dir."
            )

        local_vars = _extract_locals(locals())
        properties = properties or ModelProperties()
        properties.with_dict(local_vars)
        properties.compartment_id = properties.compartment_id or _COMPARTMENT_OCID
        auth = auth or authutil.default_signer()
        artifact_dir = _prepare_artifact_dir(artifact_dir)

        target_dir = (
            _prepare_artifact_dir()
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else artifact_dir
        )
        bucket_uri = bucket_uri or (
            artifact_dir if ObjectStorageDetails.is_oci_path(artifact_dir) else None
        )
        dsc_model = DataScienceModel.from_id(model_id)

        if not download_artifact:
            result_model = cls(
                artifact_dir=artifact_dir,
                bucket_uri=bucket_uri,
                auth=auth,
                properties=properties,
                ignore_conda_error=ignore_conda_error,
                **kwargs,
            )
            result_model.update_summary_status(
                detail=PREPARE_STATUS_GEN_SCORE_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_status(
                detail=PREPARE_STATUS_GEN_RUNTIME_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_status(
                detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_status(
                detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_status(
                detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_action(
                detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
                action="Local artifact is not available. "
                "Set load_artifact flag to True while loading the model or "
                "call .download_artifact().",
            )
            result_model.update_summary_status(
                detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.update_summary_status(
                detail=SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL,
                status=ModelState.NOTAPPLICABLE.value,
            )
            result_model.dsc_model = dsc_model
            return result_model

        dsc_model.download_artifact(
            target_dir=target_dir,
            force_overwrite=force_overwrite,
            bucket_uri=bucket_uri,
            remove_existing_artifact=remove_existing_artifact,
            auth=auth,
            region=kwargs.pop("region", None),
            timeout=kwargs.pop("timeout", None),
        )
        result_model = cls.from_model_artifact(
            uri=target_dir,
            model_file_name=model_file_name,
            artifact_dir=artifact_dir,
            auth=auth,
            force_overwrite=force_overwrite,
            properties=properties,
            ignore_conda_error=ignore_conda_error,
            **kwargs,
        )
        result_model.dsc_model = dsc_model

        result_model.update_summary_status(
            detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
            status=ModelState.DONE.value,
        )
        result_model.update_summary_action(
            detail=PREPARE_STATUS_POPULATE_METADATA_DETAIL,
            action="",
        )
        result_model.update_summary_status(
            detail=VERIFY_STATUS_LOCAL_TEST_DETAIL,
            status=ModelState.AVAILABLE.value,
        )
        result_model.update_summary_status(
            detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
            status=ModelState.AVAILABLE.value
            if not result_model.ignore_conda_error
            else ModelState.NOTAVAILABLE.value,
        )
        return result_model

    @classmethod
    def from_model_deployment(
        cls: Type[Self],
        model_deployment_id: str,
        model_file_name: str = None,
        artifact_dir: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        properties: Optional[Union[ModelProperties, Dict]] = None,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        ignore_conda_error: Optional[bool] = False,
        download_artifact: Optional[bool] = True,
        **kwargs,
    ) -> Self:
        """Loads model from model deployment.

        Parameters
        ----------
        model_deployment_id: str
            The model deployment OCID.
        model_file_name: (str, optional). Defaults to `None`.
            The name of the serialized model.
        artifact_dir: (str, optional). Defaults to `None`.
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        download_artifact: (bool, optional). Defaults to True.
            Whether to download the model pickle or checkpoints
        kwargs:
            compartment_id : (str, optional)
                Compartment OCID. If not specified, the value will be taken from the environment variables.
            timeout : (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.
            region: (str, optional). Defaults to `None`.
                The destination Object Storage bucket region.
                By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.

        Returns
        -------
        Self
            An instance of GenericModel class.
        """
        if (
            cls._PREFIX != "spark"
            and artifact_dir
            and ObjectStorageDetails.is_oci_path(artifact_dir)
        ):
            raise ValueError(
                f"Unsupported value of `artifact_dir`: {artifact_dir}. "
                "Only SparkPipelineModel framework supports object storage path as `artifact_dir`."
            )

        model_deployment = ModelDeployment.from_id(model_deployment_id)

        current_state = model_deployment.state.name.upper()
        if current_state != ModelDeploymentState.ACTIVE.name:
            logger.warning(
                "This model deployment is not in active state, you will not be able to use predict end point. "
                f"Current model deployment state: `{current_state}`"
            )

        model = cls.from_model_catalog(
            model_id=model_deployment.properties.model_id,
            model_file_name=model_file_name,
            artifact_dir=artifact_dir,
            auth=auth,
            force_overwrite=force_overwrite,
            properties=properties,
            bucket_uri=bucket_uri,
            remove_existing_artifact=remove_existing_artifact,
            ignore_conda_error=ignore_conda_error,
            download_artifact=download_artifact,
            **kwargs,
        )
        model.update_summary_status(
            detail=SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL,
            status=ModelState.AVAILABLE.value,
        )

        model.model_deployment = model_deployment
        model.update_summary_status(
            detail=DEPLOY_STATUS_DETAIL,
            status=model.model_deployment.state.name.upper(),
        )
        return model

    @class_or_instance_method
    def update_deployment(
        cls,
        model_deployment_id: str = None,
        properties: Union[ModelDeploymentProperties, dict, None] = None,
        wait_for_completion: bool = True,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        **kwargs,
    ) -> "ModelDeployment":
        """Updates a model deployment.

        You can update `model_deployment_configuration_details` and change `instance_shape` and `model_id`
        when the model deployment is in the ACTIVE lifecycle state.
        The `bandwidth_mbps` or `instance_count` can only be updated while the model deployment is in the `INACTIVE` state.
        Changes to the `bandwidth_mbps` or `instance_count` will take effect the next time
        the `ActivateModelDeployment` action is invoked on the model deployment resource.

        Examples
        --------
        >>> # Update access log id, freeform tags and description for the model deployment
        >>> model.update_deployment(
        ...     access_log={
        ...         log_id=<log_ocid>
        ...     },
        ...     description="Description for Custom Model",
        ...     freeform_tags={"key": "value"},
        ... )

        Parameters
        ----------
        model_deployment_id: str.
            The model deployment OCID. Defaults to None.
            If the method called on instance level, then `self.model_deployment.model_deployment_id` will be used.
        properties: ModelDeploymentProperties or dict
            The properties for updating the deployment.
        wait_for_completion: bool
            Flag set for whether to wait for deployment to complete before proceeding.
            Defaults to True.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).
        kwargs:
            auth: (Dict, optional). Defaults to `None`.
                The default authetication is set using `ads.set_auth` API.
                If you need to override the default, use the `ads.common.auth.api_keys` or
                `ads.common.auth.resource_principal` to create appropriate authentication signer
                and kwargs required to instantiate IdentityClient object.
            display_name: (str)
                Model deployment display name
            description: (str)
                Model deployment description
            freeform_tags: (dict)
                Model deployment freeform tags
            defined_tags: (dict)
                Model deployment defined tags

            Additional kwargs arguments.
            Can be any attribute that `ads.model.deployment.ModelDeploymentCondaRuntime`, `ads.model.deployment.ModelDeploymentContainerRuntime`
            and `ads.model.deployment.ModelDeploymentInfrastructure` accepts.

        Returns
        -------
        ModelDeployment
            An instance of ModelDeployment class.
        """
        if properties:
            warnings.warn(
                "Parameter `properties` is deprecated from GenericModel `update_deployment()` in 2.8.6 and will be removed in 3.0.0. Please use kwargs to update model deployment. "
                "Check: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/introduction.html"
            )

        if not inspect.isclass(cls):
            if cls.model_deployment:
                return cls.model_deployment.update(
                    properties=properties,
                    wait_for_completion=wait_for_completion,
                    max_wait_time=max_wait_time,
                    poll_interval=poll_interval,
                    **kwargs,
                )

        if not model_deployment_id:
            raise ValueError("Parameter `model_deployment_id` must be provided.")

        model_deployment = ModelDeployment.from_id(model_deployment_id)
        return model_deployment.update(
            properties=properties,
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
            **kwargs,
        )

    @classmethod
    def from_id(
        cls: Type[Self],
        ocid: str,
        model_file_name: str = None,
        artifact_dir: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        properties: Optional[Union[ModelProperties, Dict]] = None,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        ignore_conda_error: Optional[bool] = False,
        download_artifact: Optional[bool] = True,
        **kwargs,
    ) -> Self:
        """Loads model from model OCID or model deployment OCID.

        Parameters
        ----------
        ocid: str
            The model OCID or model deployment OCID.
        model_file_name: (str, optional). Defaults to `None`.
            The name of the serialized model.
        artifact_dir: (str, optional). Defaults to `None`.
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        download_artifact: (bool, optional). Defaults to True.
            Whether to download the model pickle or checkpoints
        kwargs:
            compartment_id : (str, optional)
                Compartment OCID. If not specified, the value will be taken from the environment variables.
            timeout : (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.

        Returns
        -------
        Self
            An instance of GenericModel class.
        """
        ocid = ocid.lower()
        if DataScienceModelType.MODEL_DEPLOYMENT in ocid:
            return cls.from_model_deployment(
                ocid,
                model_file_name=model_file_name,
                artifact_dir=artifact_dir,
                auth=auth,
                force_overwrite=force_overwrite,
                properties=properties,
                bucket_uri=bucket_uri,
                remove_existing_artifact=remove_existing_artifact,
                ignore_conda_error=ignore_conda_error,
                download_artifact=download_artifact,
                **kwargs,
            )
        elif DataScienceModelType.MODEL in ocid:
            return cls.from_model_catalog(
                ocid,
                model_file_name=model_file_name,
                artifact_dir=artifact_dir,
                auth=auth,
                force_overwrite=force_overwrite,
                properties=properties,
                bucket_uri=bucket_uri,
                remove_existing_artifact=remove_existing_artifact,
                ignore_conda_error=ignore_conda_error,
                download_artifact=download_artifact,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid OCID: {ocid}. Please provide valid model OCID or model deployment OCID."
            )

    def reload_runtime_info(self) -> None:
        """Reloads the model artifact file: `runtime.yaml`.

        Returns
        -------
        None
            Nothing.
        """
        # reload runtime.yaml
        runtime_yaml_file = os.path.join(self.artifact_dir, "runtime.yaml")
        if not utils.is_path_exists(runtime_yaml_file, auth=self.auth):
            if self.ignore_conda_error:
                return self.runtime_info
            else:
                raise FileNotFoundError(
                    f"`runtime.yaml` does not exist in {self.artifact_dir}. "
                    "Use `RuntimeInfo` class to populate it."
                )
        self.runtime_info = RuntimeInfo.from_yaml(
            uri=runtime_yaml_file, storage_options=self.auth or {}
        )

    def reload(self) -> "GenericModel":
        """Reloads the model artifact files: `score.py` and the `runtime.yaml`.

        Returns
        -------
        GenericModel
            An instance of GenericModel class.
        """
        # reload the score.py
        self.model_artifact.reload()
        # reload runtime.yaml
        self.reload_runtime_info()
        return self

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def save(
        self,
        bucket_uri: Optional[str] = None,
        defined_tags: Optional[dict] = None,
        description: Optional[str] = None,
        display_name: Optional[str] = None,
        featurestore_dataset=None,
        freeform_tags: Optional[dict] = None,
        ignore_introspection: Optional[bool] = False,
        model_version_set: Optional[Union[str, ModelVersionSet]] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        parallel_process_count: int = utils.DEFAULT_PARALLEL_PROCESS_COUNT,
        remove_existing_artifact: Optional[bool] = True,
        reload: Optional[bool] = True,
        version_label: Optional[str] = None,
        model_by_reference: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """Saves model artifacts to the model catalog.

        Parameters
        ----------
        display_name: (str, optional). Defaults to None.
            The name of the model. If a display_name is not provided in kwargs,
            randomly generated easy to remember name with timestamp will be generated,
            like 'strange-spider-2022-08-17-23:55.02'.
        description: (str, optional). Defaults to None.
            The description of the model.
        freeform_tags : Dict(str, str), Defaults to None.
            Freeform tags for the model.
        defined_tags : (Dict(str, dict(str, object)), optional). Defaults to None.
            Defined tags for the model.
        ignore_introspection: (bool, optional). Defaults to None.
            Determine whether to ignore the result of model introspection or not.
            If set to True, the save will ignore all model introspection errors.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.
        model_version_set: (Union[str, ModelVersionSet], optional). Defaults to None.
            The model version set OCID, or model version set name, or `ModelVersionSet` instance.
        version_label: (str, optional). Defaults to None.
            The model version lebel.
        featurestore_dataset: (Dataset, optional).
            The feature store dataset
        parallel_process_count: (int, optional)
            The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
        reload: (bool, optional)
            Whether to reload to check if `load_model()` works in `score.py`. Default to `True`.
        model_by_reference: (bool, optional)
            Whether model artifact is made available to Model Store by reference.
        kwargs:
            project_id: (str, optional).
                Project OCID. If not specified, the value will be taken either
                from the environment variables or model properties.
            compartment_id : (str, optional).
                Compartment OCID. If not specified, the value will be taken either
                from the environment variables or model properties.
            region: (str, optional). Defaults to `None`.
                The destination Object Storage bucket region.
                By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.
            timeout: (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.

            Also can be any attribute that `oci.data_science.models.Model` accepts.

        Raises
        ------
        RuntimeInfoInconsistencyError
            When `.runtime_info` is not synched with runtime.yaml file.

        Returns
        -------
        str
            The model id.

        Examples
        --------
        Example for saving large model artifacts (>2GB):
        >>> model.save(
        ...     bucket_uri="oci://my-bucket@my-tenancy/",
        ...     overwrite_existing_artifact=True,
        ...     remove_existing_artifact=True,
        ...     parallel_process_count=9,
        ... )

        """
        if self.model_artifact is None:
            raise ArtifactsNotAvailableError

        # Set default display_name if not specified - randomly generated easy to remember name generated
        if not display_name:
            display_name = self._random_display_name()
        # populates properties from args and kwargs. Empty values will be ignored.
        self.properties.with_dict(_extract_locals(locals()))
        self.properties.compartment_id = (
            self.properties.compartment_id or _COMPARTMENT_OCID
        )
        self.properties.project_id = self.properties.project_id or PROJECT_OCID

        # check if the runtime_info sync with the runtime.yaml.
        try:
            runtime_file_path = os.path.join(self.local_copy_dir, "runtime.yaml")
            runtime_info_from_yaml = RuntimeInfo.from_yaml(uri=runtime_file_path)
            if self.runtime_info != runtime_info_from_yaml:
                raise RuntimeInfoInconsistencyError(
                    "`.runtime_info` does not sync with runtime.yaml file. Call "
                    "`.runtime_info.save()` if you updated `runtime_info`. "
                    "Call `.reload_runtime_info()` if you updated runtime.yaml file."
                )
            # reload to check if load_model works in score.py, i.e.
            # whether the model file has been serialized, and whether it can be loaded
            # successfully.
            if reload:
                self.reload()
            else:
                logger.warning(
                    "The score.py file has not undergone testing, and this could result in deployment errors. To verify its functionality, please set `reload=True`."
                )
        except:
            if not self.ignore_conda_error:
                raise
        if not self.ignore_conda_error and not ignore_introspection:
            self._introspect()
            if self._introspect.status == TEST_STATUS.NOT_PASSED:
                msg = (
                    "Model introspection not passed. "
                    "Use `.introspect()` method to get detailed information and follow the "
                    "messages to fix it. To save model artifacts ignoring introspection "
                    "use `.save(ignore_introspection=True...)`."
                )
                self.update_summary_status(
                    detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL, status="Failed"
                )
                self.update_summary_action(
                    detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
                    action=f"Use `.introspect()` method to get detailed information.",
                )
                raise IntrospectionNotPassed(msg)
            else:
                self.update_summary_status(
                    detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL,
                    status=ModelState.DONE.value,
                )
                self.update_summary_action(
                    detail=SAVE_STATUS_INTROSPECT_TEST_DETAIL, action=""
                )

        # extract model_version_set_id from model_version_set attribute or environment
        # variables in case of saving model in context of model version set.
        model_version_set_id = _extract_model_version_set_id(model_version_set)

        if featurestore_dataset:
            dataset_details = {
                "dataset-id": featurestore_dataset.id,
                "dataset-name": featurestore_dataset.name,
            }
            self.metadata_custom.add(
                "featurestore.dataset",
                value=str(dataset_details),
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
                description="feature store dataset",
                replace=True,
            )

        self.dsc_model = (
            self.dsc_model.with_compartment_id(self.properties.compartment_id)
            .with_project_id(self.properties.project_id)
            .with_display_name(display_name)
            .with_description(description)
            .with_freeform_tags(**(freeform_tags or {}))
            .with_defined_tags(**(defined_tags or {}))
            .with_artifact(self.local_copy_dir)
            .with_model_version_set_id(model_version_set_id)
            .with_version_label(version_label)
        ).create(
            bucket_uri=bucket_uri,
            overwrite_existing_artifact=overwrite_existing_artifact,
            remove_existing_artifact=remove_existing_artifact,
            parallel_process_count=parallel_process_count,
            model_by_reference=model_by_reference,
            **kwargs,
        )

        self.update_summary_status(
            detail=SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL, status=ModelState.DONE.value
        )
        self.update_summary_status(
            detail=DEPLOY_STATUS_DETAIL, status=ModelState.AVAILABLE.value
        )
        self.model_deployment = (
            ModelDeployment()
            .with_infrastructure(ModelDeploymentInfrastructure())
            .with_runtime(ModelDeploymentContainerRuntime())
        )
        # Add the model id to the feature store dataset
        if featurestore_dataset:
            model_details = ModelDetails().with_items([self.model_id])
            featurestore_dataset.add_models(model_details)

        return self.model_id

    def _get_files(self):
        """List out all the file names under the artifact_dir.

        Returns
        -------
        List
            List of the files in the artifact_dir.
        """
        return get_files(self.artifact_dir, auth=self.auth)

    def deploy(
        self,
        wait_for_completion: Optional[bool] = True,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        deployment_instance_shape: Optional[str] = None,
        deployment_instance_subnet_id: Optional[str] = None,
        deployment_instance_count: Optional[int] = None,
        deployment_bandwidth_mbps: Optional[int] = None,
        deployment_log_group_id: Optional[str] = None,
        deployment_access_log_id: Optional[str] = None,
        deployment_predict_log_id: Optional[str] = None,
        deployment_memory_in_gbs: Optional[float] = None,
        deployment_ocpus: Optional[float] = None,
        deployment_image: Optional[str] = None,
        **kwargs: Dict,
    ) -> "ModelDeployment":
        """
        Deploys a model. The model needs to be saved to the model catalog at first. You can deploy the model
        on either conda or container runtime. The customized runtime allows you to bring your own service container.
        To deploy model on container runtime, make sure to build the container and push it to OCIR.
        For more information, see https://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm.

        Example
        -------
        >>> # This is an example to deploy model on container runtime
        >>> model = GenericModel(estimator=estimator, artifact_dir=tempfile.mkdtemp())
        >>> model.summary_status()
        >>> model.prepare(
        ...     model_file_name="toy_model.pkl",
        ...     ignore_conda_error=True, # set ignore_conda_error=True for container runtime
        ...     force_overwrite=True
        ... )
        >>> model.verify()
        >>> model.save()
        >>> model.deploy(
        ...     deployment_image="iad.ocir.io/<namespace>/<image>:<tag>",
        ...     entrypoint=["python", "/opt/ds/model/deployed_model/api.py"],
        ...     server_port=5000,
        ...     health_check_port=5000,
        ...     environment_variables={"key":"value"}
        ... )

        Parameters
        ----------
        wait_for_completion : (bool, optional). Defaults to True.
            Flag set for whether to wait for deployment to complete before proceeding.
        display_name: (str, optional). Defaults to None.
            The name of the model. If a display_name is not provided in kwargs,
            a randomly generated easy to remember name with timestamp will be generated,
            like 'strange-spider-2022-08-17-23:55.02'.
        description: (str, optional). Defaults to None.
            The description of the model.
        deployment_instance_shape: (str, optional). Default to `VM.Standard2.1`.
            The shape of the instance used for deployment.
        deployment_instance_subnet_id: (str, optional). Default to None.
            The subnet id of the instance used for deployment.
        deployment_instance_count: (int, optional). Defaults to 1.
            The number of instance used for deployment.
        deployment_bandwidth_mbps: (int, optional). Defaults to 10.
            The bandwidth limit on the load balancer in Mbps.
        deployment_memory_in_gbs: (float, optional). Defaults to None.
            Specifies the size of the memory of the model deployment instance in GBs.
        deployment_ocpus: (float, optional). Defaults to None.
            Specifies the ocpus count of the model deployment instance.
        deployment_log_group_id: (str, optional). Defaults to None.
            The oci logging group id. The access log and predict log share the same log group.
        deployment_access_log_id: (str, optional). Defaults to None.
            The access log OCID for the access logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        deployment_predict_log_id: (str, optional). Defaults to None.
            The predict log OCID for the predict logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        deployment_image: (str, optional). Defaults to None.
            The OCIR path of docker container image. Required for deploying model on container runtime.
        kwargs:
            project_id: (str, optional).
                Project OCID. If not specified, the value will be taken from the environment variables.
            compartment_id : (str, optional).
                Compartment OCID. If not specified, the value will be taken from the environment variables.
            max_wait_time : (int, optional). Defaults to 1200 seconds.
                Maximum amount of time to wait in seconds.
                Negative implies infinite wait time.
            poll_interval : (int, optional). Defaults to 10 seconds.
                Poll interval in seconds.
            freeform_tags: (Dict[str, str], optional). Defaults to None.
                Freeform tags of the model deployment.
            defined_tags: (Dict[str, dict[str, object]], optional). Defaults to None.
                Defined tags of the model deployment.
            image_digest: (str, optional). Defaults to None.
                The digest of docker container image.
            cmd: (List, optional). Defaults to empty.
                The command line arguments for running docker container image.
            entrypoint: (List, optional). Defaults to empty.
                The entrypoint for running docker container image.
            server_port: (int, optional). Defaults to 8080.
                The server port for docker container image.
            health_check_port: (int, optional). Defaults to 8080.
                The health check port for docker container image.
            deployment_mode: (str, optional). Defaults to HTTPS_ONLY.
                The deployment mode. Allowed values are: HTTPS_ONLY and STREAM_ONLY.
            input_stream_ids: (List, optional). Defaults to empty.
                The input stream ids. Required for STREAM_ONLY mode.
            output_stream_ids: (List, optional). Defaults to empty.
                The output stream ids. Required for STREAM_ONLY mode.
            environment_variables: (Dict, optional). Defaults to empty.
                The environment variables for model deployment.

            Also can be any keyword argument for initializing the `ads.model.deployment.ModelDeploymentProperties`.
            See `ads.model.deployment.ModelDeploymentProperties()` for details.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance.

        Raises
        ------
        ValueError
            If `model_id` is not specified.
        """
        # Set default display_name if not specified - randomly generated easy to remember name generated
        if not display_name:
            display_name = utils.get_random_name_for_resource()
        # populates properties from args and kwargs. Empty values will be ignored.
        override_properties = _extract_locals(locals())
        # clears out project_id and compartment_id from kwargs, to prevent passing
        # these params to the deployment via kwargs.
        kwargs.pop("project_id", None)
        kwargs.pop("compartment_id", None)

        max_wait_time = kwargs.pop("max_wait_time", DEFAULT_WAIT_TIME)
        poll_interval = kwargs.pop("poll_interval", DEFAULT_POLL_INTERVAL)

        # GenericModel itself has a ModelDeployment instance. When calling deploy(),
        # if there are parameters passed in they will override this ModelDeployment instance,
        # otherwise the properties of the ModelDeployment instance will be applied for deployment.
        existing_infrastructure = self.model_deployment.infrastructure
        existing_runtime = self.model_deployment.runtime
        property_dict = ModelProperties(
            compartment_id=existing_infrastructure.compartment_id
            or self.properties.compartment_id
            or _COMPARTMENT_OCID,
            project_id=existing_infrastructure.project_id
            or self.properties.project_id
            or PROJECT_OCID,
            deployment_instance_shape=existing_infrastructure.shape_name
            or self.properties.deployment_instance_shape
            or MODEL_DEPLOYMENT_INSTANCE_SHAPE,
            deployment_instance_count=existing_infrastructure.replica
            or self.properties.deployment_instance_count
            or MODEL_DEPLOYMENT_INSTANCE_COUNT,
            deployment_bandwidth_mbps=existing_infrastructure.bandwidth_mbps
            or self.properties.deployment_bandwidth_mbps
            or MODEL_DEPLOYMENT_BANDWIDTH_MBPS,
            deployment_ocpus=existing_infrastructure.shape_config_details.get(
                "ocpus", None
            )
            or self.properties.deployment_ocpus
            or MODEL_DEPLOYMENT_INSTANCE_OCPUS,
            deployment_memory_in_gbs=existing_infrastructure.shape_config_details.get(
                "memoryInGBs", None
            )
            or self.properties.deployment_memory_in_gbs
            or MODEL_DEPLOYMENT_INSTANCE_MEMORY_IN_GBS,
            deployment_log_group_id=existing_infrastructure.log_group_id
            or self.properties.deployment_log_group_id,
            deployment_access_log_id=existing_infrastructure.access_log.get(
                "log_id", None
            )
            or self.properties.deployment_access_log_id,
            deployment_predict_log_id=existing_infrastructure.predict_log.get(
                "log_id", None
            )
            or self.properties.deployment_predict_log_id,
            deployment_image=getattr(existing_runtime, "image", None)
            or self.properties.deployment_image,
            deployment_instance_subnet_id=existing_infrastructure.subnet_id
            or self.properties.deployment_instance_subnet_id,
        ).to_dict()

        property_dict.update(override_properties)
        self.properties.with_dict(property_dict)

        if not self.model_id:
            raise ValueError(
                "The model needs to be saved to the Model Catalog "
                "before it can be deployed."
            )

        if (
            self.properties.deployment_access_log_id
            or self.properties.deployment_predict_log_id
        ) and not self.properties.deployment_log_group_id:
            raise ValueError(
                "`deployment_log_group_id` needs to be specified. "
                "`deployment_access_log_id` and `deployment_predict_log_id` "
                "cannot be used without `deployment_log_group_id`."
            )

        if not self.properties.compartment_id:
            raise ValueError("`compartment_id` has to be provided.")
        if not self.properties.project_id:
            raise ValueError("`project_id` has to be provided.")
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_compartment_id(self.properties.compartment_id)
            .with_project_id(self.properties.project_id)
            .with_bandwidth_mbps(self.properties.deployment_bandwidth_mbps)
            .with_shape_name(self.properties.deployment_instance_shape)
            .with_replica(self.properties.deployment_instance_count)
            .with_subnet_id(self.properties.deployment_instance_subnet_id)
        )

        web_concurrency = (
            kwargs.pop("web_concurrency", None)
            or existing_infrastructure.web_concurrency
        )
        if web_concurrency:
            infrastructure.with_web_concurrency(web_concurrency)

        if infrastructure.shape_name.endswith("Flex"):
            infrastructure.with_shape_config_details(
                ocpus=self.properties.deployment_ocpus,
                memory_in_gbs=self.properties.deployment_memory_in_gbs,
            )

        # specifies the access log id
        if self.properties.deployment_access_log_id:
            infrastructure.with_access_log(
                log_group_id=self.properties.deployment_log_group_id,
                log_id=self.properties.deployment_access_log_id,
            )

        # specifies the predict log id
        if self.properties.deployment_predict_log_id:
            infrastructure.with_predict_log(
                log_group_id=self.properties.deployment_log_group_id,
                log_id=self.properties.deployment_predict_log_id,
            )

        environment_variables = (
            kwargs.pop("environment_variables", {}) or existing_runtime.env
        )
        deployment_mode = (
            kwargs.pop("deployment_mode", None)
            or existing_runtime.deployment_mode
            or ModelDeploymentMode.HTTPS
        )

        runtime = None
        if self.properties.deployment_image:
            image_digest = kwargs.pop("image_digest", None) or getattr(
                existing_runtime, "image_digest", None
            )
            cmd = kwargs.pop("cmd", []) or getattr(existing_runtime, "cmd", [])
            entrypoint = kwargs.pop("entrypoint", []) or getattr(
                existing_runtime, "entrypoint", []
            )
            server_port = kwargs.pop("server_port", None) or getattr(
                existing_runtime, "server_port", None
            )
            health_check_port = kwargs.pop("health_check_port", None) or getattr(
                existing_runtime, "health_check_port", None
            )
            runtime = (
                ModelDeploymentContainerRuntime()
                .with_image(self.properties.deployment_image)
                .with_image_digest(image_digest)
                .with_cmd(cmd)
                .with_entrypoint(entrypoint)
                .with_server_port(server_port)
                .with_health_check_port(health_check_port)
                .with_deployment_mode(deployment_mode)
                .with_model_uri(self.model_id)
                .with_env(environment_variables)
            )
        else:
            runtime = (
                ModelDeploymentCondaRuntime()
                .with_env(environment_variables)
                .with_deployment_mode(deployment_mode)
                .with_model_uri(self.model_id)
            )

        if deployment_mode == ModelDeploymentMode.STREAM:
            input_stream_ids = (
                kwargs.pop("input_stream_ids", []) or existing_runtime.input_stream_ids
            )
            output_stream_ids = (
                kwargs.pop("output_stream_ids", [])
                or existing_runtime.output_stream_ids
            )
            if not (input_stream_ids and output_stream_ids):
                raise ValueError(
                    "Parameter `input_stream_ids` and `output_stream_ids` need to be provided for `STREAM_ONLY` mode."
                )

            runtime.with_input_stream_ids(input_stream_ids)
            runtime.with_output_stream_ids(output_stream_ids)

        freeform_tags = (
            kwargs.pop("freeform_tags", {}) or self.model_deployment.freeform_tags
        )
        defined_tags = (
            kwargs.pop("defined_tags", {}) or self.model_deployment.defined_tags
        )

        model_deployment = (
            ModelDeployment()
            .with_display_name(display_name or self.model_deployment.display_name)
            .with_description(description or self.model_deployment.description)
            .with_defined_tags(**defined_tags)
            .with_freeform_tags(**freeform_tags)
            .with_infrastructure(infrastructure)
            .with_runtime(runtime)
        )

        self.model_deployment = model_deployment.deploy(
            wait_for_completion=wait_for_completion,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )
        self.update_summary_status(
            detail=DEPLOY_STATUS_DETAIL,
            status=self.model_deployment.state.name.upper(),
        )
        return self.model_deployment

    def prepare_save_deploy(
        self,
        inference_conda_env: str = None,
        inference_python_version: str = None,
        training_conda_env: str = None,
        training_python_version: str = None,
        model_file_name: str = None,
        as_onnx: bool = False,
        initial_types: List[Tuple] = None,
        force_overwrite: bool = False,
        namespace: str = CONDA_BUCKET_NS,
        use_case_type: str = None,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        training_script_path: str = None,
        training_id: str = _TRAINING_RESOURCE_ID,
        ignore_pending_changes: bool = True,
        max_col_num: int = DATA_SCHEMA_MAX_COL_NUM,
        ignore_conda_error: bool = False,
        model_display_name: Optional[str] = None,
        model_description: Optional[str] = None,
        model_freeform_tags: Optional[dict] = None,
        model_defined_tags: Optional[dict] = None,
        ignore_introspection: Optional[bool] = False,
        wait_for_completion: Optional[bool] = True,
        deployment_display_name: Optional[str] = None,
        deployment_description: Optional[str] = None,
        deployment_instance_shape: Optional[str] = None,
        deployment_instance_subnet_id: Optional[str] = None,
        deployment_instance_count: Optional[int] = None,
        deployment_bandwidth_mbps: Optional[int] = None,
        deployment_log_group_id: Optional[str] = None,
        deployment_access_log_id: Optional[str] = None,
        deployment_predict_log_id: Optional[str] = None,
        deployment_memory_in_gbs: Optional[float] = None,
        deployment_ocpus: Optional[float] = None,
        deployment_image: Optional[str] = None,
        bucket_uri: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
        model_version_set: Optional[Union[str, ModelVersionSet]] = None,
        version_label: Optional[str] = None,
        model_by_reference: Optional[bool] = False,
        **kwargs: Dict,
    ) -> "ModelDeployment":
        """Shortcut for prepare, save and deploy steps.

        Parameters
        ----------
        inference_conda_env: (str, optional). Defaults to None.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
        inference_python_version: (str, optional). Defaults to None.
            Python version which will be used in deployment.
        training_conda_env: (str, optional). Defaults to None.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
            If `training_conda_env` is not provided, `training_conda_env` will
            use the same value of `training_conda_env`.
        training_python_version: (str, optional). Defaults to None.
            Python version used during training.
        model_file_name: (str, optional). Defaults to `None`.
            Name of the serialized model.
        as_onnx: (bool, optional). Defaults to False.
            Whether to serialize as onnx model.
        initial_types: (list[Tuple], optional).
            Defaults to None. Only used for SklearnModel, LightGBMModel and XGBoostModel.
            Each element is a tuple of a variable name and a type.
            Check this link http://onnx.ai/sklearn-onnx/api_summary.html#id2 for
            more explanation and examples for `initial_types`.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files.
        namespace: (str, optional).
            Namespace of region. This is used for identifying which region the service pack
            is from when you pass a slug to inference_conda_env and training_conda_env.
        use_case_type: str
            The use case type of the model. Use it through UserCaseType class or string provided in `UseCaseType`. For
            example, use_case_type=UseCaseType.BINARY_CLASSIFICATION or use_case_type="binary_classification". Check
            with UseCaseType class to see all supported types.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema.
        y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of output data that will be used to generate output schema.
        training_script_path: str. Defaults to None.
            Training script path.
        training_id: (str, optional). Defaults to value from environment variables.
            The training OCID for model. Can be notebook session or job OCID.
        ignore_pending_changes: bool. Defaults to False.
            whether to ignore the pending changes in the git.
        max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
            Do not generate the input schema if the input has more than this
            number of features(columns).
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        model_display_name: (str, optional). Defaults to None.
            The name of the model. If a model_display_name is not provided in kwargs,
            a randomly generated easy to remember name with timestamp will be generated,
            like 'strange-spider-2022-08-17-23:55.02'.
        model_description: (str, optional). Defaults to None.
            The description of the model.
        model_freeform_tags : Dict(str, str), Defaults to None.
            Freeform tags for the model.
        model_defined_tags : (Dict(str, dict(str, object)), optional). Defaults to None.
            Defined tags for the model.
        ignore_introspection: (bool, optional). Defaults to None.
            Determine whether to ignore the result of model introspection or not.
            If set to True, the save will ignore all model introspection errors.
        wait_for_completion : (bool, optional). Defaults to True.
            Flag set for whether to wait for deployment to complete before proceeding.
        deployment_display_name: (str, optional). Defaults to None.
            The name of the model deployment. If a deployment_display_name is not provided in kwargs,
            a randomly generated easy to remember name with timestamp will be generated,
            like 'strange-spider-2022-08-17-23:55.02'.
        description: (str, optional). Defaults to None.
            The description of the model.
        deployment_instance_shape: (str, optional). Default to `VM.Standard2.1`.
            The shape of the instance used for deployment.
        deployment_instance_subnet_id: (str, optional). Default to None.
            The subnet id of the instance used for deployment.
        deployment_instance_count: (int, optional). Defaults to 1.
            The number of instance used for deployment.
        deployment_bandwidth_mbps: (int, optional). Defaults to 10.
            The bandwidth limit on the load balancer in Mbps.
        deployment_log_group_id: (str, optional). Defaults to None.
            The oci logging group id. The access log and predict log share the same log group.
        deployment_access_log_id: (str, optional). Defaults to None.
            The access log OCID for the access logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        deployment_predict_log_id: (str, optional). Defaults to None.
            The predict log OCID for the predict logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        deployment_memory_in_gbs: (float, optional). Defaults to None.
            Specifies the size of the memory of the model deployment instance in GBs.
        deployment_ocpus: (float, optional). Defaults to None.
            Specifies the ocpus count of the model deployment instance.
        deployment_image: (str, optional). Defaults to None.
            The OCIR path of docker container image. Required for deploying model on container runtime.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        model_version_set: (Union[str, ModelVersionSet], optional). Defaults to None.
            The Model version set OCID, or name, or `ModelVersionSet` instance.
        version_label: (str, optional). Defaults to None.
            The model version lebel.
        model_by_reference: (bool, optional)
            Whether model artifact is made available to Model Store by reference.
        kwargs:
            impute_values: (dict, optional).
                The dictionary where the key is the column index(or names is accepted
                for pandas dataframe) and the value is the impute value for the corresponding column.
            project_id: (str, optional).
                Project OCID. If not specified, the value will be taken either
                from the environment variables or model properties.
            compartment_id : (str, optional).
                Compartment OCID. If not specified, the value will be taken either
                from the environment variables or model properties.
            image_digest: (str, optional). Defaults to None.
                The digest of docker container image.
            cmd: (List, optional). Defaults to empty.
                The command line arguments for running docker container image.
            entrypoint: (List, optional). Defaults to empty.
                The entrypoint for running docker container image.
            server_port: (int, optional). Defaults to 8080.
                The server port for docker container image.
            health_check_port: (int, optional). Defaults to 8080.
                The health check port for docker container image.
            deployment_mode: (str, optional). Defaults to HTTPS_ONLY.
                The deployment mode. Allowed values are: HTTPS_ONLY and STREAM_ONLY.
            input_stream_ids: (List, optional). Defaults to empty.
                The input stream ids. Required for STREAM_ONLY mode.
            output_stream_ids: (List, optional). Defaults to empty.
                The output stream ids. Required for STREAM_ONLY mode.
            environment_variables: (Dict, optional). Defaults to empty.
                The environment variables for model deployment.
            timeout: (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.
            max_wait_time : (int, optional). Defaults to 1200 seconds.
                Maximum amount of time to wait in seconds.
                Negative implies infinite wait time.
            poll_interval : (int, optional). Defaults to 10 seconds.
                Poll interval in seconds.
            freeform_tags: (Dict[str, str], optional). Defaults to None.
                Freeform tags of the model deployment.
            defined_tags: (Dict[str, dict[str, object]], optional). Defaults to None.
                Defined tags of the model deployment.
            region: (str, optional). Defaults to `None`.
                The destination Object Storage bucket region.
                By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.

            Also can be any keyword argument for initializing the
            `ads.model.deployment.ModelDeploymentProperties`.
            See `ads.model.deployment.ModelDeploymentProperties()` for details.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance.

        Raises
        ------
        FileExistsError
            If files already exist but `force_overwrite` is False.
        ValueError
            If `inference_python_version` is not provided,
            but also cannot be found through manifest file.
        """
        locals_dict = _extract_locals(locals())
        locals_dict.pop("training_id", None)
        self.properties.with_dict(locals_dict)

        self.prepare(
            inference_conda_env=self.properties.inference_conda_env,
            inference_python_version=self.properties.inference_python_version,
            training_conda_env=self.properties.training_conda_env,
            training_python_version=self.properties.training_python_version,
            model_file_name=model_file_name,
            as_onnx=as_onnx,
            initial_types=initial_types,
            force_overwrite=force_overwrite,
            namespace=namespace,
            use_case_type=use_case_type,
            X_sample=X_sample,
            y_sample=y_sample,
            training_script_path=self.properties.training_script_path,
            training_id=self.properties.training_id,
            ignore_pending_changes=ignore_pending_changes,
            max_col_num=max_col_num,
            ignore_conda_error=ignore_conda_error,
            impute_values=kwargs.pop("impute_values", None),
        )
        # Set default model_display_name if not specified - randomly generated easy to remember name generated
        if not model_display_name:
            model_display_name = utils.get_random_name_for_resource()

        self.save(
            display_name=model_display_name,
            description=model_description,
            freeform_tags=model_freeform_tags,
            defined_tags=model_defined_tags,
            ignore_introspection=ignore_introspection,
            compartment_id=self.properties.compartment_id,
            project_id=self.properties.project_id,
            timeout=kwargs.pop("timeout", None),
            bucket_uri=bucket_uri,
            overwrite_existing_artifact=overwrite_existing_artifact,
            remove_existing_artifact=remove_existing_artifact,
            model_version_set=model_version_set,
            version_label=version_label,
            region=kwargs.pop("region", None),
            model_by_reference=model_by_reference,
        )
        # Set default deployment_display_name if not specified - randomly generated easy to remember name generated
        if not deployment_display_name:
            deployment_display_name = utils.get_random_name_for_resource()

        self.deploy(
            wait_for_completion=wait_for_completion,
            display_name=deployment_display_name,
            description=deployment_description,
            deployment_instance_shape=self.properties.deployment_instance_shape,
            deployment_instance_subnet_id=self.properties.deployment_instance_subnet_id,
            deployment_instance_count=self.properties.deployment_instance_count,
            deployment_bandwidth_mbps=self.properties.deployment_bandwidth_mbps,
            deployment_log_group_id=self.properties.deployment_log_group_id,
            deployment_access_log_id=self.properties.deployment_access_log_id,
            deployment_predict_log_id=self.properties.deployment_predict_log_id,
            deployment_memory_in_gbs=self.properties.deployment_memory_in_gbs,
            deployment_ocpus=self.properties.deployment_ocpus,
            deployment_image=deployment_image,
            kwargs=kwargs,
        )
        return self.model_deployment

    def predict(
        self,
        data: Any = None,
        auto_serialize_data: bool = False,
        local: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Returns prediction of input data run against the model deployment endpoint.

        Examples
        --------
        >>> uri = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        >>> prediction = model.predict(image=uri)['prediction']

        >>> # examples on storage options
        >>> prediction = model.predict(
        ...        image="oci://<bucket>@<tenancy>/myimage.png",
        ...        storage_options=ads.auth.default_signer()
        ... )['prediction']

        Parameters
        ----------
        data: Any
            Data for the prediction for onnx models, for local serialization
            method, data can be the data types that each framework support.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Defauls to `False` for GenericModel, and `True` for other frameworks.
            `data` required to be json serializable if `auto_serialize_data=False`.
            If `auto_serialize_data` set to True, data will be serialized before sending to model deployment endpoint.
        local: bool.
            Whether to invoke the prediction locally. Default to False.
        kwargs:
            content_type: str, used to indicate the media type of the resource.
            image: PIL.Image Object or uri for the image.
               A valid string path for image file can be local path, http(s), oci, s3, gs.
            storage_options: dict
               Passed to `fsspec.open` for a particular storage connection.
               Please see `fsspec` (https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open) for more details.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the predicted values.

        Raises
        ------
        NotActiveDeploymentError
            If model deployment process was not started or not finished yet.
        ValueError
            If model is not deployed yet or the endpoint information is not available.
        """
        if local:
            return self.verify(
                data=data, auto_serialize_data=auto_serialize_data, **kwargs
            )

        if not (self.model_deployment and self.model_deployment.url):
            raise ValueError(
                "Error invoking the remote endpoint as the model is not "
                "deployed yet or the endpoint information is not available. "
                "Use `deploy()` method to start model deployment. "
                "If you intend to invoke inference using locally available "
                "model artifact, set parameter `local=True`"
            )

        current_state = self.model_deployment.state.name.upper()
        if current_state != ModelDeploymentState.ACTIVE.name:
            raise NotActiveDeploymentError(current_state)

        data = self._handle_input_data(data, auto_serialize_data, **kwargs)
        prediction = self.model_deployment.predict(
            data=data,
            serializer=self.get_data_serializer(),
            **kwargs,
        )

        self.update_summary_status(
            detail=PREDICT_STATUS_CALL_ENDPOINT_DETAIL, status=ModelState.DONE.value
        )
        return prediction

    def summary_status(self) -> pd.DataFrame:
        """A summary table of the current status.

        Returns
        -------
        pd.DataFrame
            The summary stable of the current status.
        """
        if (
            not self.ignore_conda_error
            and self.model_file_name
            and not os.path.exists(
                os.path.join(self.artifact_dir, self.model_file_name)
            )
        ):
            self.update_summary_action(
                detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                action=f"Model is not automatically serialized. Serialize the model as `{self.model_file_name}` and save to the {self.artifact_dir}.",
            )
            self.update_summary_status(
                detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                status=ModelState.NEEDSACTION.value,
            )
        else:
            self.update_summary_action(
                detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL, action=""
            )
            if (
                ModelState.NEEDSACTION.value
                in self._summary_status.df.loc[
                    self._summary_status.df["Details"]
                    == PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                    "Status",
                ].values
            ):
                self.update_summary_status(
                    detail=PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                    status=ModelState.DONE.value,
                )
        if (
            self._summary_status.df.loc[
                self._summary_status.df["Details"] == PREPARE_STATUS_GEN_SCORE_DETAIL,
                "Actions Needed",
            ].values
            != ""
        ):
            try:
                self.model_artifact.reload()
                self.update_summary_action(
                    detail=PREPARE_STATUS_GEN_SCORE_DETAIL, action=""
                )
            except:
                pass

        if self.model_deployment:
            self.update_summary_status(
                detail=DEPLOY_STATUS_DETAIL,
                status=self.model_deployment.state.name.upper(),
            )

            if self.model_deployment.state == ModelDeploymentState.ACTIVE:
                self.update_summary_status(
                    detail=PREDICT_STATUS_CALL_ENDPOINT_DETAIL,
                    status=ModelState.AVAILABLE.value,
                )
            elif (
                self.model_deployment.state
                and self.model_deployment.state != ModelDeploymentState.ACTIVE
            ):
                self.update_summary_status(
                    detail=PREDICT_STATUS_CALL_ENDPOINT_DETAIL,
                    status=ModelState.NOTAVAILABLE.value,
                )

        return self._summary_status.df.set_index(["Step", "Status", "Details"])

    def update_summary_status(self, detail: str, status: str):
        """Update the status in the summary table.

        Parameters
        ----------
        detail: (str)
            value of the detail in the details column of the summary status table. Used to locate which row to update.
        status: (str)
            new status to be updated for the row specified by detail.


        Returns
        -------
        None
        """
        self._summary_status.update_status(detail=detail, status=status)

    def update_summary_action(self, detail: str, action: str):
        """Update the actions needed from the user in the summary table.

        Parameters
        ----------
        detail: (str)
            value of the detail in the details column of the summary status table. Used to locate which row to update.
        action: (str)
            new action to be updated for the row specified by detail.

        Returns
        -------
        None
        """
        self._summary_status.update_action(detail=detail, action=action)

    def delete_deployment(self, wait_for_completion: bool = True) -> None:
        """Deletes the current deployment.

        Parameters
        ----------
        wait_for_completion: (bool, optional). Defaults to `True`.
            Whether to wait till completion.

        Returns
        -------
        None

        Raises
        ------
        ValueError: if there is not deployment attached yet.
        """
        if not self.model_deployment:
            raise ValueError("Use `deploy()` method to start model deployment.")
        self.model_deployment.delete(wait_for_completion=wait_for_completion)

    def restart_deployment(
        self,
        max_wait_time: int = DEFAULT_WAIT_TIME,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> "ModelDeployment":
        """Restarts the current deployment.

        Parameters
        ----------
        max_wait_time : (int, optional). Defaults to 1200 seconds.
            Maximum amount of time to wait for activate or deactivate in seconds.
            Total amount of time to wait for restart deployment is twice as the value.
            Negative implies infinite wait time.
        poll_interval : (int, optional). Defaults to 10 seconds.
            Poll interval in seconds.

        Returns
        -------
        ModelDeployment
            The ModelDeployment instance.
        """
        if not self.model_deployment:
            raise ValueError("Use `deploy()` method to start model deployment.")
        logger.info(
            f"Deactivating model deployment {self.model_deployment.model_deployment_id}."
        )
        self.model_deployment.deactivate(
            max_wait_time=max_wait_time, poll_interval=poll_interval
        )
        logger.info(
            f"Model deployment {self.model_deployment.model_deployment_id} has successfully been deactivated."
        )
        logger.info(
            f"Activating model deployment {self.model_deployment.model_deployment_id}."
        )
        self.model_deployment.activate(
            max_wait_time=max_wait_time, poll_interval=poll_interval
        )
        logger.info(
            f"Model deployment {self.model_deployment.model_deployment_id} has successfully been activated."
        )
        return self.model_deployment

    @class_or_instance_method
    def delete(
        cls,
        model_id: Optional[str] = None,
        delete_associated_model_deployment: Optional[bool] = False,
        delete_model_artifact: Optional[bool] = False,
        artifact_dir: Optional[str] = None,
        **kwargs: Dict,
    ) -> None:
        """
        Deletes a model from Model Catalog.

        Parameters
        ----------
        model_id: (str, optional). Defaults to None.
            The model OCID to be deleted.
            If the method called on instance level, then `self.model_id` will be used.
        delete_associated_model_deployment: (bool, optional). Defaults to `False`.
            Whether associated model deployments need to be deleted or not.
        delete_model_artifact: (bool, optional). Defaults to `False`.
            Whether associated model artifacts need to be deleted or not.
        artifact_dir: (str, optional). Defaults to `None`
            The local path to the model artifacts folder.
            If the method called on instance level,
            the `self.artifact_dir` will be used by default.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `model_id` not provided.
        """
        if not inspect.isclass(cls):
            model_id = model_id or cls.model_id
            artifact_dir = artifact_dir or cls.artifact_dir

        if not model_id:
            raise ValueError("The `model_id` must be provided.")
        if delete_model_artifact and not artifact_dir:
            raise ValueError("The `artifact_dir` must be provided.")

        DataScienceModel.from_id(model_id).delete(
            delete_associated_model_deployment=delete_associated_model_deployment
        )

        if delete_model_artifact:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def upload_artifact(
        self,
        uri: str,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        parallel_process_count: int = utils.DEFAULT_PARALLEL_PROCESS_COUNT,
    ) -> None:
        """Uploads model artifacts to the provided `uri`.
        The artifacts will be zipped before uploading.

        Parameters
        ----------
        uri: str
            The destination location for the model artifacts, which can be a local path or
            OCI object storage URI. Examples:

            >>> upload_artifact(uri="/some/local/folder/")
            >>> upload_artifact(uri="oci://bucket@namespace/prefix/")

        auth: (Dict, optional). Defaults to `None`.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        force_overwrite: bool
            Overwrite target_dir if exists.
        parallel_process_count: (int, optional)
            The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
        """
        if not uri:
            raise ValueError("The `uri` must be provided.")

        if not self.artifact_dir:
            raise ValueError(
                "The model artifacts not found. "
                "Use `prepare()` method to prepare model artifacts."
            )

        if not os.path.basename(uri) and self.model_id:
            uri = os.path.join(uri, f"{self.model_id}.zip")

        tmp_artifact_zip_path = None
        progressbar_description = f"Uploading an artifact ZIP archive to {uri}."
        try:
            # Zip artifacts
            tmp_artifact_zip_path = zip_artifact(self.artifact_dir)
            # Upload artifacts to the provided destination
            if ObjectStorageDetails.is_oci_path(
                uri
            ) and ObjectStorageDetails.is_valid_uri(uri):
                utils.upload_to_os(
                    src_uri=tmp_artifact_zip_path,
                    dst_uri=uri,
                    auth=auth,
                    parallel_process_count=parallel_process_count,
                    progressbar_description=progressbar_description,
                )
            else:
                utils.copy_file(
                    uri_src=tmp_artifact_zip_path,
                    uri_dst=uri,
                    auth=auth,
                    force_overwrite=force_overwrite,
                    progressbar_description=progressbar_description,
                )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to upload model artifact to the given Object Storage path `{uri}`."
                f"See Exception: {ex}"
            )
        finally:
            if tmp_artifact_zip_path:
                os.remove(tmp_artifact_zip_path)

    def update(self, **kwargs) -> "GenericModel":
        """Updates model metadata in the Model Catalog.
        Updates only metadata information. The model artifacts are immutable and cannot be updated.

        Parameters
        ----------
        kwargs
            display_name: (str, optional). Defaults to None.
                The name of the model.
            description: (str, optional). Defaults to None.
                The description of the model.
            freeform_tags : Dict(str, str), Defaults to None.
                Freeform tags for the model.
            defined_tags : (Dict(str, dict(str, object)), optional). Defaults to None.
                Defined tags for the model.
            version_label: (str, optional). Defaults to None.
                The model version lebel.

            Additional kwargs arguments.
            Can be any attribute that `oci.data_science.models.Model` accepts.

        Returns
        -------
        GenericModel
            An instance of `GenericModel` (self).

        Raises
        ------
        ValueError
            if model not saved to the Model Catalog.
        """
        if not self.model_id:
            raise ValueError(
                "Use `save()` method to save a model to the Model Catalog."
            )

        self.dsc_model = (
            self.dsc_model.with_display_name(
                kwargs.pop("display_name", self.dsc_model.display_name)
            )
            .with_description(kwargs.pop("description", self.dsc_model.description))
            .with_freeform_tags(
                **(
                    kwargs.pop("freeform_tags", self.dsc_model.freeform_tags or {})
                    or {}
                )
            )
            .with_defined_tags(
                **(kwargs.pop("defined_tags", self.dsc_model.defined_tags or {}) or {})
            )
            .with_version_label(
                kwargs.pop("version_label", self.dsc_model.version_label)
            )
            .update(**kwargs)
        )

        return self


class ModelState(Enum):
    DONE = "Done"
    AVAILABLE = "Available"
    NOTAVAILABLE = "Not Available"
    NEEDSACTION = "Needs Action"
    NOTAPPLICABLE = "Not Applicable"


class SummaryStatus:
    """SummaryStatus class which track the status of the Model frameworks."""

    def __init__(self):
        summary_data = [
            [INITIATE_STATUS_NAME, INITIATE_STATUS_DETAIL, ModelState.DONE.value, ""],
            [
                PREPARE_STATUS_NAME,
                PREPARE_STATUS_GEN_RUNTIME_DETAIL,
                ModelState.AVAILABLE.value,
                "",
            ],
            [
                PREPARE_STATUS_NAME,
                PREPARE_STATUS_GEN_SCORE_DETAIL,
                ModelState.AVAILABLE.value,
                "",
            ],
            [
                PREPARE_STATUS_NAME,
                PREPARE_STATUS_SERIALIZE_MODEL_DETAIL,
                ModelState.AVAILABLE.value,
                "",
            ],
            [
                PREPARE_STATUS_NAME,
                PREPARE_STATUS_POPULATE_METADATA_DETAIL,
                ModelState.AVAILABLE.value,
                "",
            ],
            [
                VERIFY_STATUS_NAME,
                VERIFY_STATUS_LOCAL_TEST_DETAIL,
                ModelState.NOTAVAILABLE.value,
                "",
            ],
            [
                SAVE_STATUS_NAME,
                SAVE_STATUS_INTROSPECT_TEST_DETAIL,
                ModelState.NOTAVAILABLE.value,
                "",
            ],
            [
                SAVE_STATUS_NAME,
                SAVE_STATUS_UPLOAD_ARTIFACT_DETAIL,
                ModelState.NOTAVAILABLE.value,
                "",
            ],
            [
                DEPLOY_STATUS_NAME,
                DEPLOY_STATUS_DETAIL,
                ModelState.NOTAVAILABLE.value,
                "",
            ],
            [
                PREDICT_STATUS_NAME,
                PREDICT_STATUS_CALL_ENDPOINT_DETAIL,
                ModelState.NOTAVAILABLE.value,
                "",
            ],
        ]
        self.df = pd.DataFrame(
            summary_data, columns=["Step", "Details", "Status", "Actions Needed"]
        )

    def update_status(self, detail: str, status: str) -> None:
        """Updates the status of the summary status table of the corresponding detail.

        Parameters
        ----------
        detail: (str)
            value of the detail in the Details column. Used to locate which row to update.
        status: (str)
            new status to be updated for the row specified by detail.

        Returns
        -------
        None
            Nothing.
        """
        self.df.loc[self.df["Details"] == detail, "Status"] = status

    def update_action(self, detail: str, action: str) -> None:
        """Updates the action of the summary status table of the corresponding detail.

        Parameters
        ----------
        detail: (str)
            Value of the detail in the Details column. Used to locate which row to update.
        action: (str)
            new action to be updated for the row specified by detail.

        Returns
        -------
        None
            Nothing.
        """
        self.df.loc[
            self.df["Details"] == detail,
            "Actions Needed",
        ] = action


class FrameworkSpecificModel(GenericModel):
    def verify(
        self,
        data: Any = None,
        reload_artifacts: bool = True,
        auto_serialize_data: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Test if deployment works in local environment.

        Examples
        --------
        >>> uri = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        >>> prediction = model.verify(image=uri)['prediction']

        >>> # examples on storage options
        >>> prediction = model.verify(
        ...        image="oci://<bucket>@<tenancy>/myimage.png",
        ...        storage_options=ads.auth.default_signer()
        ... )['prediction']

        Parameters
        ----------
        data: Any
            Data used to test if deployment works in local environment.
        reload_artifacts: bool. Defaults to True.
            Whether to reload artifacts or not.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Defauls to `False` for GenericModel, and `True` for other frameworks.
            `data` required to be json serializable if `auto_serialize_data=False`.
            if `auto_serialize_data` set to True, data will be serialized before sending to model deployment endpoint.
        kwargs:
            content_type: str, used to indicate the media type of the resource.
            image: PIL.Image Object or uri for the image.
               A valid string path for image file can be local path, http(s), oci, s3, gs.
            storage_options: dict
               Passed to `fsspec.open` for a particular storage connection.
               Please see `fsspec` (https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open) for more details.

        Returns
        -------
        Dict
            A dictionary which contains prediction results.
        """
        return super().verify(
            data=data,
            reload_artifacts=reload_artifacts,
            auto_serialize_data=auto_serialize_data,
            **kwargs,
        )

    def predict(
        self, data: Any = None, auto_serialize_data: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """Returns prediction of input data run against the model deployment endpoint.

        Examples
        --------
        >>> uri = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        >>> prediction = model.predict(image=uri)['prediction']

        >>> # examples on storage options
        >>> prediction = model.predict(
        ...        image="oci://<bucket>@<tenancy>/myimage.png",
        ...        storage_options=ads.auth.default_signer()
        ... )['prediction']

        Parameters
        ----------
        data: Any
            Data for the prediction for onnx models, for local serialization
            method, data can be the data types that each framework support.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Defauls to `False` for GenericModel, and `True` for other frameworks.
            `data` required to be json serializable if `auto_serialize_data=False`.
            If `auto_serialize_data` set to True, data will be serialized before sending to model deployment endpoint.
        kwargs:
            content_type: str, used to indicate the media type of the resource.
            image: PIL.Image Object or uri for the image.
               A valid string path for image file can be local path, http(s), oci, s3, gs.
            storage_options: dict
               Passed to `fsspec.open` for a particular storage connection.
               Please see `fsspec` (https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open) for more details.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the predicted values.

        Raises
        ------
        NotActiveDeploymentError
            If model deployment process was not started or not finished yet.
        ValueError
            If `data` is empty or not JSON serializable.
        """
        return super().predict(
            data=data, auto_serialize_data=auto_serialize_data, **kwargs
        )
