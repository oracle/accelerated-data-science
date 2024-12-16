#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from ads.model.extractor.embedding_onnx_extractor import EmbeddingONNXExtractor
from ads.model.generic_model import FrameworkSpecificModel


class EmbeddingONNXModel(FrameworkSpecificModel):
    """EmbeddingONNXModel class for embedding onnx model.

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
    framework: str
        "embedding_onnx", the framework name of the model.
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
    >>> import os
    >>> import shutil
    >>> from ads.model import EmbeddingONNXModel
    >>> from huggingface_hub import snapshot_download

    >>> local_dir=tempfile.mkdtemp()
    >>> # download sentence-transformers/all-MiniLM-L6-v2 from huggingface
    >>> snapshot_download(
    ...     repo_id="sentence-transformers/all-MiniLM-L6-v2",
    ...     local_dir=local_dir
    ... )

    >>> # copy all files from local_dir to artifact_dir
    >>> artifact_dir = tempfile.mkdtemp()
    >>> for root, dirs, files in os.walk(local_dir):
    >>>     for file in files:
    >>>         src_path = os.path.join(root, file)
    >>>         shutil.copy(src_path, artifact_dir)

    >>> model = EmbeddingONNXModel(artifact_dir=artifact_dir)
    >>> model.summary_status()
    >>> model.prepare(
    ...     inference_conda_env="onnxruntime_p311_gpu_x86_64",
    ...     inference_python_version="3.11",
    ...     model_file_name="model.onnx",
    ...     force_overwrite=True
    ... )
    >>> model.summary_status()
    >>> model.verify(
    ...     {
    ...         "input": ['What are activation functions?', 'What is Deep Learning?'],
    ...         "model": "sentence-transformers/all-MiniLM-L6-v2"
    ...     },
    ... )
    >>> model.summary_status()
    >>> model.save(display_name="sentence-transformers/all-MiniLM-L6-v2")
    >>> model.summary_status()
    >>> model.deploy(
    ...    display_name="all-MiniLM-L6-v2 Embedding deployment",
    ...    deployment_instance_shape="VM.Standard.E4.Flex",
    ...    deployment_ocpus=20,
    ...    deployment_memory_in_gbs=256,
    ... )
    >>> model.predict(
    ...     {
    ...         "input": ['What are activation functions?', 'What is Deep Learning?'],
    ...         "model": "sentence-transformers/all-MiniLM-L6-v2"
    ...     },
    ... )
    >>> # Uncomment the line below to delete the model and the associated model deployment
    >>> # model.delete(delete_associated_model_deployment = True)
    """

    def __init__(
        self,
        artifact_dir: str | None = None,
        auth: Dict | None = None,
        serialize: bool = False,
        **kwargs: dict,
    ):
        """
        Initiates a EmbeddingONNXModel instance.

        Parameters
        ----------
        artifact_dir: str
            Directory for generate artifact.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        serialize: bool
            Whether to serialize the model to pkl file by default.
            Required as `False` for embedding onnx model.

        Returns
        -------
        EmbeddingONNXModel
            EmbeddingONNXModel instance.

        Examples
        --------
        >>> import tempfile
        >>> import os
        >>> import shutil
        >>> from ads.model import EmbeddingONNXModel
        >>> from huggingface_hub import snapshot_download

        >>> local_dir=tempfile.mkdtemp()
        >>> # download sentence-transformers/all-MiniLM-L6-v2 from huggingface
        >>> snapshot_download(
        ...     repo_id="sentence-transformers/all-MiniLM-L6-v2",
        ...     local_dir=local_dir
        ... )

        >>> # copy all files from subdirectory to artifact_dir
        >>> artifact_dir = tempfile.mkdtemp()
        >>> for root, dirs, files in os.walk(local_dir):
        >>>     for file in files:
        >>>         src_path = os.path.join(root, file)
        >>>         shutil.copy(src_path, artifact_dir)

        >>> model = EmbeddingONNXModel(artifact_dir=artifact_dir)
        >>> model.summary_status()
        >>> model.prepare(
        ...     inference_conda_env="onnxruntime_p311_gpu_x86_64",
        ...     inference_python_version="3.11",
        ...     model_file_name="model.onnx",
        ...     force_overwrite=True
        ... )
        >>> model.summary_status()
        >>> model.verify(
        ...     {
        ...         "input": ['What are activation functions?', 'What is Deep Learning?'],
        ...         "model": "sentence-transformers/all-MiniLM-L6-v2"
        ...     },
        ... )
        >>> model.summary_status()
        >>> model.save(display_name="sentence-transformers/all-MiniLM-L6-v2")
        >>> model.summary_status()
        >>> model.deploy(
        ...    display_name="all-MiniLM-L6-v2 Embedding deployment",
        ...    deployment_instance_shape="VM.Standard.E4.Flex",
        ...    deployment_ocpus=20,
        ...    deployment_memory_in_gbs=256,
        ... )
        >>> model.predict(
        ...     {
        ...         "input": ['What are activation functions?', 'What is Deep Learning?'],
        ...         "model": "sentence-transformers/all-MiniLM-L6-v2"
        ...     },
        ... )
        >>> # Uncomment the line below to delete the model and the associated model deployment
        >>> # model.delete(delete_associated_model_deployment = True)
        """
        super().__init__(
            artifact_dir=artifact_dir,
            auth=auth,
            serialize=serialize,
            **kwargs,
        )

        self._extractor = EmbeddingONNXExtractor()
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    def verify(
        self, data=None, reload_artifacts=True, auto_serialize_data=False, **kwargs
    ):
        """Test if embedding onnx model deployment works in local environment.

        Examples
        --------
        >>> data = {
        ...     "input": ['What are activation functions?', 'What is Deep Learning?'],
        ...     "model": "sentence-transformers/all-MiniLM-L6-v2"
        ... }
        >>> prediction = model.verify(data)

        Parameters
        ----------
        data: Any
            Data used to test if deployment works in local environment.
        reload_artifacts: bool. Defaults to True.
            Whether to reload artifacts or not.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Required as `False` for embedding onnx model.
            Input `data` must be json serializable.
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
        if auto_serialize_data:
            raise ValueError(
                "ADS will not auto serialize `data` for embedding onnx model. "
                "Input json serializable `data` and set `auto_serialize_data` as False."
            )

        return super().verify(
            data=data,
            reload_artifacts=reload_artifacts,
            auto_serialize_data=auto_serialize_data,
            **kwargs,
        )

    def predict(self, data=None, auto_serialize_data=False, **kwargs):
        """Returns prediction of input data run against the embedding onnx model deployment endpoint.

        Examples
        --------
        >>> data = {
        ...     "input": ['What are activation functions?', 'What is Deep Learning?'],
        ...     "model": "sentence-transformers/all-MiniLM-L6-v2"
        ... }
        >>> prediction = model.predict(data)

        Parameters
        ----------
        data: Any
            Data for the prediction for model deployment.
        auto_serialize_data: bool.
            Whether to auto serialize input data. Required as `False` for embedding onnx model.
            Input `data` must be json serializable.
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
        """
        if auto_serialize_data:
            raise ValueError(
                "ADS will not auto serialize `data` for embedding onnx model. "
                "Input json serializable `data` and set `auto_serialize_data` as False."
            )

        return super().predict(
            data=data, auto_serialize_data=auto_serialize_data, **kwargs
        )
