#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - HuggingFace
"""
from abc import ABC, abstractmethod
import json
import os
import shutil
import tempfile
from unittest.mock import patch
import transformers

import cloudpickle
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import pipeline  # mocked

from ads.model.framework.huggingface_model import HuggingFacePipelineModel
from ads.model.serde.model_serializer import HuggingFaceModelSerializer


def create_image():
    arr = np.zeros([686, 960, 3], dtype=np.uint8)
    arr[:, :100] = [255, 128, 0]
    arr[:, 100:] = [0, 0, 255]
    return Image.fromarray(arr)


@pytest.fixture
def image_data():
    return create_image()


class Config:
    use_pretrained_backbone = True

    def save_pretrained(self, save_directory):
        with open(
            os.path.join(save_directory, "preprocessor_config.json"), mode="w"
        ) as f:
            f.write("something")
        with open(os.path.join(save_directory, "config.json"), mode="w") as f:
            f.write("something")

    def to_dict(self):
        return {"hyperparameter": "value"}


class Model:
    config = Config()


class FakePipeline(transformers.pipelines.base.Pipeline):
    def __init__(self, task, model):
        self.task = task
        self.model = model

    @abstractmethod
    def __call__(self, images):
        pass

    def save_pretrained(self, save_directory):
        with open(os.path.join(save_directory, "pytorch_model.bin"), mode="w") as f:
            f.write("something")
        self.model.config.save_pretrained(save_directory)

    def _forward(self):
        pass

    def _sanitize_parameters(self):
        pass

    def postprocess(self):
        pass

    def preprocess(self):
        pass


class FakePipelineImage(FakePipeline):
    def __call__(self, images):
        return {"prediction": create_image()}


class FakePipelineImageList(FakePipeline):
    def __call__(self, images):
        return [{"prediction": create_image()}]


class FakePipelineImageTensor(FakePipeline):
    def __call__(self, images):
        return torch.tensor([0])


class FakePipelineMultipleInputs(FakePipeline):
    def __call__(self, images, candidate_labels):
        return [{"prediction": "result"}]


@pytest.fixture
def fake_pipeline():
    fakepipeline = FakePipelineImage("fake_task", Model())
    return fakepipeline


@pytest.fixture
def fake_pipeline_list():
    fakepipeline = FakePipelineImageList("fake_task", Model())
    return fakepipeline


@pytest.fixture
def fake_pipeline_tensor():
    fakepipeline = FakePipelineImageTensor("fake_task", Model())
    return fakepipeline


@pytest.fixture
def fake_pipeline_multiple_inputs():
    fakepipeline = FakePipelineMultipleInputs("fake_task", Model())
    return fakepipeline


class TestHuggingFaceModelSerializer:
    """Unittests for HuggingFaceModelSerializer class"""

    def setup_class(cls):
        cls.tmp_model_dir = tempfile.mkdtemp()

    def test_serialize(self, fake_pipeline):

        os.makedirs(self.tmp_model_dir, exist_ok=True)
        HuggingFaceModelSerializer().serialize(fake_pipeline, self.tmp_model_dir)
        assert os.path.exists(os.path.join(self.tmp_model_dir, "pytorch_model.bin"))
        assert os.path.exists(
            os.path.join(self.tmp_model_dir, "preprocessor_config.json")
        )
        assert os.path.exists(os.path.join(self.tmp_model_dir, "config.json"))

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir, ignore_errors=True)


class TestHuggingFacePipelineModel:
    """Unittests for the HuggingFacePipelineModel class."""

    def setup_class(cls):
        cls.tmp_model_dir = tempfile.mkdtemp()
        os.makedirs(cls.tmp_model_dir, exist_ok=True)

        cls.image = create_image()
        cls.image_bytes = cloudpickle.dumps(cls.image)

        cls.multi_inputs = {"images": cls.image, "candidate_labels": ["a"]}
        cls.multi_inputs_bytes = cloudpickle.dumps(cls.multi_inputs)
        cls.conda = "oci://fake_bucket@fake_namespace/fake_conda"

    def test_serialize(self, fake_pipeline):
        """
        Test serialize.
        """

        model = HuggingFacePipelineModel(fake_pipeline, artifact_dir=self.tmp_model_dir)
        model.serialize_model(force_overwrite=True)
        assert os.path.exists(os.path.join(self.tmp_model_dir, "pytorch_model.bin"))
        assert os.path.exists(
            os.path.join(self.tmp_model_dir, "preprocessor_config.json")
        )
        assert os.path.exists(os.path.join(self.tmp_model_dir, "config.json"))

    def test_serialize_as_onnx_warning(self, fake_pipeline):
        """
        Test serialize as_onnx=True gives warning.
        """

        with pytest.raises(
            NotImplementedError,
            match="HuggingFace Pipeline to onnx conversion is not supported.",
        ):
            model = HuggingFacePipelineModel(
                fake_pipeline, artifact_dir=self.tmp_model_dir
            )
            model.serialize_model(as_onnx=True, force_overwrite=True)

    def test_init(self, fake_pipeline):

        model = HuggingFacePipelineModel(fake_pipeline, artifact_dir=self.tmp_model_dir)
        assert model.algorithm == "FakePipelineImage"
        assert model.framework == "transformers"
        assert model.task == "fake_task"
        assert model.model_input_serializer.name == "cloudpickle"
        assert model.model_save_serializer.name == "huggingface"

    @patch("transformers.pipeline")
    def test_prepare_verify_image(self, pipeline_mock, fake_pipeline):
        """test prepare verify function where pipeline input data is an image and output is dictionary with image in it."""
        pipeline_mock.return_value = fake_pipeline
        model = HuggingFacePipelineModel(fake_pipeline, artifact_dir=self.tmp_model_dir)

        model.prepare(
            inference_conda_env=self.conda,
            inference_python_version="3.8",
            force_overwrite=True,
        )
        assert model.model_file_name == self.tmp_model_dir

        prediction_from_image = model.verify(self.image)
        prediction_from_bytes = model.verify(
            self.image_bytes, auto_serialize_data=False
        )

        json.dumps(prediction_from_image)
        json.dumps(prediction_from_bytes)
        assert prediction_from_bytes == prediction_from_image

    @patch("transformers.pipeline")
    def test_prepare_verify_list_image(self, pipeline_mock, fake_pipeline_list):
        """test prepare verify function where pipeline input data is an image and output is list of dictionary with image in it."""
        pipeline_mock.return_value = fake_pipeline_list
        model = HuggingFacePipelineModel(
            fake_pipeline_list, artifact_dir=self.tmp_model_dir
        )

        model.prepare(
            inference_conda_env=self.conda,
            inference_python_version="3.8",
            force_overwrite=True,
        )
        assert model.model_file_name == self.tmp_model_dir

        prediction_from_image = model.verify(self.image)
        prediction_from_bytes = model.verify(
            self.image_bytes, auto_serialize_data=False
        )

        json.dumps(prediction_from_image)
        json.dumps(prediction_from_bytes)
        assert prediction_from_bytes == prediction_from_image
        assert isinstance(prediction_from_image["prediction"], list)

    @patch("transformers.pipeline")
    def test_prepare_verify_tensor(self, pipeline_mock, fake_pipeline_tensor):
        """test prepare verify function where pipeline input data is an image and output is torch.Tensor."""

        pipeline_mock.return_value = fake_pipeline_tensor
        model = HuggingFacePipelineModel(
            fake_pipeline_tensor, artifact_dir=self.tmp_model_dir
        )

        model.prepare(
            inference_conda_env=self.conda,
            inference_python_version="3.8",
            force_overwrite=True,
        )
        assert model.model_file_name == self.tmp_model_dir

        prediction_from_image = model.verify(self.image)
        prediction_from_bytes = model.verify(
            self.image_bytes, auto_serialize_data=False
        )

        json.dumps(prediction_from_image)
        json.dumps(prediction_from_bytes)
        assert prediction_from_bytes == prediction_from_image
        assert isinstance(prediction_from_image["prediction"], list)

    @patch("transformers.pipeline")
    def test_prepare_verify_multiple_inputs(
        self, pipeline_mock, fake_pipeline_multiple_inputs
    ):
        """test prepare verify function where pipeline input data is an image and output is torch.Tensor."""

        pipeline_mock.return_value = fake_pipeline_multiple_inputs
        model = HuggingFacePipelineModel(
            fake_pipeline_multiple_inputs, artifact_dir=self.tmp_model_dir
        )

        model.prepare(
            inference_conda_env=self.conda,
            inference_python_version="3.8",
            force_overwrite=True,
        )
        assert model.model_file_name == self.tmp_model_dir

        prediction_from_image = model.verify(self.multi_inputs)
        prediction_from_bytes = model.verify(
            self.multi_inputs_bytes, auto_serialize_data=False
        )

        json.dumps(prediction_from_image)
        json.dumps(prediction_from_bytes)
        assert prediction_from_bytes == prediction_from_image
        assert isinstance(prediction_from_image["prediction"], list)

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir, ignore_errors=True)
