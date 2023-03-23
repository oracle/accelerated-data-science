#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL

from ads.model.extractor.huggingface_extractor import HuggingFaceExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.model_serializer import HuggingFaceSerializerType
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.serde.model_serializer import ModelSerializerType
from ads.model.serde.common import SERDE


class HuggingFacePipelineModel(FrameworkSpecificModel):
    """HuggingFacePipelineModel class for estimators from HuggingFace framework.

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
        A trained HuggingFace Pipeline using transformers.
    framework: str
        "transformers", the framework name of the model.
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
    >>> # Image Classification
    >>> from transformers import pipeline
    >>> import tempfile
    >>> import PIL.Image
    >>> import ads
    >>> import requests
    >>> import cloudpickle
    >>> ## Download image data
    >>> image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    >>> image = PIL.Image.open(requests.get(image_link, stream=True).raw)
    >>> image_bytes = cloudpickle.dumps(image) # convert image to bytes
    >>> ## Download a pretrained model
    >>> vision_classifier = pipeline(model="google/vit-base-patch16-224")
    >>> preds = vision_classifier(images=image)
    >>> ## Initiate a HuggingFacePipelineModel instance
    >>> vision_model = HuggingFacePipelineModel(vision_classifier, artifact_dir=tempfile.mkdtemp())
    >>> ## Prepare
    >>> vision_model.prepare(inference_conda_env="pytorch110_p38_cpu_v1", force_overwrite=True)
    >>> ## Verify
    >>> vision_model.verify(image)
    >>> vision_model.verify(image_bytes)
    >>> ## Save
    >>> vision_model.save()
    >>> ## Deploy
    >>> log_group_id = "<log_group_id>"
    >>> log_id = "<log_id>"
    >>> vision_model.deploy(deployment_bandwidth_mbps=1000,
    ...                wait_for_completion=False,
    ...                deployment_log_group_id = log_group_id,
    ...                deployment_access_log_id = log_id,
    ...                deployment_predict_log_id = log_id)
    >>> ## Predict from endpoint
    >>> vision_model.predict(image)
    >>> vision_model.predict(image_bytes)
    >>> ### Invoke the model
    >>> auth = ads.common.auth.default_signer()['signer']
    >>> endpoint = vision_model.model_deployment.url + "/predict"
    >>> headers = {"Content-Type": "application/octet-stream"}
    >>> requests.post(endpoint, data=image_bytes, auth=auth, headers=headers).json()

    Examples
    --------
    >>> # Image Segmentation
    >>> from transformers import pipeline
    >>> import tempfile
    >>> import PIL.Image
    >>> import ads
    >>> import requests
    >>> import cloudpickle
    >>> ## Download image data
    >>> image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    >>> image = PIL.Image.open(requests.get(image_link, stream=True).raw)
    >>> image_bytes = cloudpickle.dumps(image) # convert image to bytes
    >>> ## Download pretrained model
    >>> segmenter = pipeline(task="image-segmentation")
    >>> preds = segmenter(image)
    >>> ## Initiate a HuggingFacePipelineModel instance
    >>> segmentation_model = HuggingFacePipelineModel(segmenter, artifact_dir=empfile.mkdtemp())
    >>> ## Prepare
    >>> conda = "oci://bucket@namespace/path/to/conda/pack"
    >>> python_version = "3.8"
    >>> segmentation_model.prepare(inference_conda_env=conda, inference_python_version = python_version, force_overwrite=True)
    >>> ## Verify
    >>> segmentation_model.verify(data=image)
    >>> segmentation_model.verify(data=image_bytes)
    >>> ## Save
    >>> segmentation_model.save()
    >>> log_group_id = "<log_group_id>"
    >>> log_id = "<log_id>"
    >>> ## Deploy
    >>> segmentation_model.deploy(deployment_bandwidth_mbps=1000,
                    wait_for_completion=False,
                    deployment_log_group_id = log_group_id,
                    deployment_access_log_id = log_id,
                    deployment_predict_log_id = log_id)
    >>> ## Predict from endpoint
    >>> segmentation_model.predict(image)
    >>> segmentation_model.predict(image_bytes)
    >>> ## Invoke the model
    >>> auth = ads.common.auth.default_signer()['signer']

    >>> endpoint = segmentation_model.model_deployment.url + "/predict"
    >>> headers = {"Content-Type": "application/octet-stream"}
    >>> requests.post(endpoint, data=image_bytes, auth=auth, headers=headers).json()

    Examples
    --------
    >>> # Zero Shot Image Classification
    >>> from transformers import pipeline
    >>> import tempfile
    >>> import PIL.Image
    >>> import ads
    >>> import requests
    >>> import cloudpickle
    >>> ## Download the image data
    >>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    >>> image = PIL.Image.open(requests.get(image_link, stream=True).raw)
    >>> image_bytes = cloudpickle.dumps(image)
    >>> ## Download a pretrained model
    >>> classifier = pipeline(model="openai/clip-vit-large-patch14")
    >>> classifier(
            images=image,
            candidate_labels=["animals", "humans", "landscape"],
        )
    >>> ## Initiate a HuggingFacePipelineModel instance
    >>> zero_shot_image_classification_model = HuggingFacePipelineModel(classifier, artifact_dir=empfile.mkdtemp())
    >>> conda = "oci://bucket@namespace/path/to/conda/pack"
    >>> python_version = "3.8"
    >>> ## Prepare
    >>> zero_shot_image_classification_model.prepare(inference_conda_env=conda, inference_python_version = python_version, force_overwrite=True)
    >>> data = {"images": image, "candidate_labels": ["animals", "humans", "landscape"]}
    >>> body = cloudpickle.dumps(data) # convert image to bytes
    >>> ## Verify
    >>> zero_shot_image_classification_model.verify(data=data)
    >>> zero_shot_image_classification_model.verify(data=body)
    >>> ## Save
    >>> zero_shot_image_classification_model.save()
    >>> ## Deploy
    >>> log_group_id = "<log_group_id>"
    >>> log_id = "<log_id>"
    >>> zero_shot_image_classification_model.deploy(deployment_bandwidth_mbps=1000,
                    wait_for_completion=False,
                    deployment_log_group_id = log_group_id,
                    deployment_access_log_id = log_id,
                    deployment_predict_log_id = log_id)
    >>> ## Predict from endpoint
    >>> zero_shot_image_classification_model.predict(image)
    >>> zero_shot_image_classification_model.predict(body)
    >>> ### Invoke the model
    >>> auth = ads.common.auth.default_signer()['signer']
    >>> endpoint = zero_shot_image_classification_model.model_deployment.url + "/predict"
    >>> headers = {"Content-Type": "application/octet-stream"}
    >>> requests.post(endpoint, data=body, auth=auth, headers=headers).json()
    """

    _PREFIX = "huggingface"
    model_save_serializer_type = HuggingFaceSerializerType

    @runtime_dependency(
        module="transformers", install_from=OptionalDependency.HUGGINGFACE
    )
    def __init__(
        self,
        estimator: Callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = model_save_serializer_type.HUGGINGFACE,
        model_input_serializer: Optional[SERDE] = ModelSerializerType.CLOUDPICKLE,
        **kwargs,
    ):
        """
        Initiates a HuggingFacePipelineModel instance.

        Parameters
        ----------
        estimator: Callable
            HuggingFacePipeline Model
        artifact_dir: str
            Directory for generate artifact.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
            For more details, check https://accelerated-data-science.readthedocs.io/en/latest/ads.model.html#module-ads.model.model_properties.
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
        HuggingFacePipelineModel
            HuggingFacePipelineModel instance.

        Examples
        --------
        >>> from transformers import pipeline
        >>> import tempfile
        >>> import PIL.Image
        >>> import ads
        >>> import requests
        >>> import cloudpickle
        >>> ## download the image
        >>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
        >>> image = PIL.Image.open(requests.get(image_link, stream=True).raw)
        >>> image_bytes = cloudpickle.dumps(image)
        >>> ## download the pretrained model
        >>> classifier = pipeline(model="openai/clip-vit-large-patch14")
        >>> classifier(
                images=image,
                candidate_labels=["animals", "humans", "landscape"],
            )
        >>> ## Initiate a HuggingFacePipelineModel instance
        >>> zero_shot_image_classification_model = HuggingFacePipelineModel(classifier, artifact_dir=empfile.mkdtemp())
        >>> ## Prepare a model artifact
        >>> conda = "oci://bucket@namespace/path/to/conda/pack"
        >>> python_version = "3.8"
        >>> zero_shot_image_classification_model.prepare(inference_conda_env=conda, inference_python_version = python_version, force_overwrite=True)
        >>> ## Test data
        >>> data = {"images": image, "candidate_labels": ["animals", "humans", "landscape"]}
        >>> body = cloudpickle.dumps(data) # convert image to bytes
        >>> ## Verify
        >>> zero_shot_image_classification_model.verify(data=data)
        >>> zero_shot_image_classification_model.verify(data=body)
        >>> ## Save
        >>> zero_shot_image_classification_model.save()
        >>> ## Deploy
        >>> log_group_id = "<log_group_id>"
        >>> log_id = "<log_id>"
        >>> zero_shot_image_classification_model.deploy(deployment_bandwidth_mbps=100,
                        wait_for_completion=False,
                        deployment_log_group_id = log_group_id,
                        deployment_access_log_id = log_id,
                        deployment_predict_log_id = log_id)
        >>> zero_shot_image_classification_model.predict(image)
        >>> zero_shot_image_classification_model.predict(body)
        >>> ### Invoke the model by sending bytes
        >>> auth = ads.common.auth.default_signer()['signer']
        >>> endpoint = zero_shot_image_classification_model.model_deployment.url + "/predict"
        >>> headers = {"Content-Type": "application/octet-stream"}
        >>> requests.post(endpoint, data=body, auth=auth, headers=headers).json()
        """
        if not isinstance(estimator, transformers.pipelines.base.Pipeline):
            raise TypeError(
                f"{str(type(estimator))} is not supported in HuggingFacePipelineModel."
            )
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            model_save_serializer=model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )
        self._extractor = HuggingFaceExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter
        self.task = self.estimator.task
        self._score_args["task"] = self.estimator.task

    def _handle_model_file_name(
        self, as_onnx: bool = False, model_file_name: str = None
    ):
        """
        The artifact directory to store model files.

        Parameters
        ----------
        as_onnx: bool. Defaults to False
            If set as True, it will be ignored as onnx conversion is not supported.
        model_file_name: str
            Will be ignored as huggingface pipeline requires to folder to store the model
            files and those files will be stored at the artifact directory.

        Returns
        -------
        str
            The artifact directory.
        """
        return self.artifact_dir

    def serialize_model(
        self,
        as_onnx: bool = False,
        force_overwrite: bool = False,
        X_sample: Optional[Union[Dict, str, List, PIL.Image.Image]] = None,
        **kwargs,
    ) -> None:
        """
        Serialize and save HuggingFace model using model specific method.

        Parameters
        ----------
        as_onnx: (bool, optional). Defaults to False.
            If set as True, convert into ONNX model.
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[Dict, str, List, PIL.Image.Image]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect onnx_args.

        Returns
        -------
        None
            Nothing.
        """

        if as_onnx:
            raise NotImplementedError(
                "HuggingFace Pipeline to onnx conversion is not supported."
            )

        super().serialize_model(
            as_onnx=False,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )
