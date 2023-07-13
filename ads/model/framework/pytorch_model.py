#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.model.extractor.pytorch_extractor import PytorchExtractor
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.model_serializer import PyTorchModelSerializerType
from ads.model.common.utils import (
    DEPRECATE_AS_ONNX_WARNING,
    DEPRECATE_USE_TORCH_SCRIPT_WARNING,
)
from ads.model.serde.common import SERDE

ONNX_MODEL_FILE_NAME = "model.onnx"
PYTORCH_MODEL_FILE_NAME = "model.pt"


class PyTorchModel(FrameworkSpecificModel):
    """PyTorchModel class for estimators from Pytorch framework.

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
        A trained pytorch estimator/model using Pytorch.
    framework: str
        "pytorch", the framework name of the model.
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
    properties: ModelProperties
        ModelProperties object required to save and deploy model.
        For more details, check https://accelerated-data-science.readthedocs.io/en/latest/ads.model.html#module-ads.model.model_properties.
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
    from_model_artifact(uri, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from the specified folder, or zip/tar archive.
    from_model_catalog(model_id, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from model catalog.
    introspect(...)
        Runs model introspection.
    predict(data, ...)
        Returns prediction of input data run against the model deployment endpoint.
    prepare(..., **kwargs)
        Prepare and save the score.py, serialized model and runtime.yaml file.
    reload(...)
        Reloads the model artifact files: `score.py` and the `runtime.yaml`.
    save(..., **kwargs)
        Saves model artifacts to the model catalog.
    summary_status(...)
        Gets a summary table of the current status.
    verify(data, ...)
        Tests if deployment works in local environment.

    Examples
    --------
    >>> torch_model = PyTorchModel(estimator=torch_estimator,
    ... artifact_dir=tmp_model_dir)
    >>> inference_conda_env = "generalml_p37_cpu_v1"

    >>> torch_model.prepare(inference_conda_env=inference_conda_env, force_overwrite=True)
    >>> torch_model.reload()
    >>> torch_model.verify(...)
    >>> torch_model.save()
    >>> model_deployment = torch_model.deploy(wait_for_completion=False)
    >>> torch_model.predict(...)
    """

    _PREFIX = "pytorch"
    model_save_serializer_type = PyTorchModelSerializerType

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def __init__(
        self,
        estimator: callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = model_save_serializer_type.TORCH,
        model_input_serializer: Optional[SERDE] = None,
        **kwargs,
    ):
        """
        Initiates a PyTorchModel instance.

        Parameters
        ----------
        estimator: callable
            Any model object generated by pytorch framework
        artifact_dir: str
            artifact directory to store the files needed for deployment.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize data.

        Returns
        -------
        PyTorchModel
            PyTorchModel instance.
        """
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            model_save_serializer=model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )
        self._extractor = PytorchExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter
        self.version = torch.__version__

    def serialize_model(
        self,
        as_onnx: bool = False,
        force_overwrite: bool = False,
        X_sample: Optional[
            Union[
                Dict,
                str,
                List,
                Tuple,
                np.ndarray,
                pd.core.series.Series,
                pd.core.frame.DataFrame,
            ]
        ] = None,
        use_torch_script: bool = None,
        **kwargs,
    ) -> None:
        """
        Serialize and save Pytorch model using ONNX or model specific method.

        Parameters
        ----------
        as_onnx: (bool, optional). Defaults to False.
            If set as True, convert into ONNX model.
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect onnx_args.
        use_torch_script:  (bool, optional). Defaults to None (If the default value has not been changed, it will be set as `False`).
            If set as `True`, the model will be serialized as a TorchScript program. Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format for more details.
            If set as `False`, it will only save the trained model’s learned parameters, and the score.py
            need to be modified to construct the model class instance first. Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended for more details.
        **kwargs: optional params used to serialize pytorch model to onnx,
        including the following:
            onnx_args: (tuple or torch.Tensor), default to None
            Contains model inputs such that model(onnx_args) is a valid
            invocation of the model. Can be structured either as: 1) ONLY A
            TUPLE OF ARGUMENTS; 2) A TENSOR; 3) A TUPLE OF ARGUMENTS ENDING
            WITH A DICTIONARY OF NAMED ARGUMENTS
            input_names: (List[str], optional). Names to assign to the input
            nodes of the graph, in order.
            output_names: (List[str], optional). Names to assign to the output nodes of the graph, in order.
            dynamic_axes: (dict, optional), default to None. Specify axes of tensors as dynamic (i.e. known only at run-time).

        Returns
        -------
        None
            Nothing.
        """
        if use_torch_script is None:
            logger.warning(
                "In future the models will be saved in TorchScript format by default. Currently saving it using torch.save method."
                "Set `use_torch_script` as `True` to serialize the model as a TorchScript program by `torch.jit.save()` "
                "and loaded using `torch.jit.load()` in score.py. "
                "You don't need to modify `load_model()` in score.py to load the model."
                "Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format for more details."
                "Set `use_torch_script` as `False` to save only the model parameters."
                "The model class instance must be constructed before "
                "loading parameters in the predict function of score.py."
                "Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended for more details."
            )
            use_torch_script = False

        if as_onnx and use_torch_script:
            raise ValueError("You can only save Pytorch model into one format.")

        if as_onnx:
            logger.warning(DEPRECATE_AS_ONNX_WARNING)
            self.set_model_save_serializer(self.model_save_serializer_type.ONNX)

        if use_torch_script:
            logger.warning(DEPRECATE_USE_TORCH_SCRIPT_WARNING)
            self.set_model_save_serializer(self.model_save_serializer_type.TORCHSCRIPT)

        super().serialize_model(
            as_onnx=as_onnx,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )

    def _to_tensor(self, data):
        try:
            import torchvision.transforms as transforms

            convert_tensor = transforms.ToTensor()
            data = convert_tensor(data)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"The `torchvision` module was not found. Please run "
                f"`pip install {OptionalDependency.PYTORCH}`."
            )
        return data
