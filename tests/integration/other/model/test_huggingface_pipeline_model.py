#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - HuggingFace
"""

import os
import shutil
import tempfile

import cloudpickle
import json
import PIL
import pytest
from transformers import pipeline

from ads.model.framework.huggingface_model import HuggingFacePipelineModel
from ads.model.serde.model_serializer import HuggingFaceModelSerializer


@pytest.fixture
def segmenter_pipeline():
    return pipeline(
        task="image-segmentation", model="facebook/maskformer-swin-tiny-coco"
    )


class TestHuggingFaceModelSerializer:
    """Unittests for HuggingFaceModelSerializer class"""

    def setup_class(cls):
        cls.tmp_model_dir = tempfile.mkdtemp()

    def test_serialize(self, segmenter_pipeline):
        os.makedirs(self.tmp_model_dir, exist_ok=True)
        HuggingFaceModelSerializer().serialize(segmenter_pipeline, self.tmp_model_dir)
        assert os.path.exists(os.path.join(self.tmp_model_dir, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(self.tmp_model_dir, "config.json"))

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir, ignore_errors=True)


class TestHuggingFacePipelineModel:
    """Unittests for the HuggingFacePipelineModel class."""

    def setup_class(cls):
        cls.tmp_model_dir = tempfile.mkdtemp()
        os.makedirs(cls.tmp_model_dir, exist_ok=True)
        #
        cls.image_url = "tests/integration/other/model/image_files/dog.jpeg"
        cls.image = PIL.Image.open(cls.image_url)
        cls.image_bytes = cloudpickle.dumps(cls.image)
        cls.conda = "oci://fake_bucket@fake_namespace/fake_conda"

    def test_serialize(self, segmenter_pipeline):
        """
        Test serialize.
        """
        model = HuggingFacePipelineModel(
            segmenter_pipeline, artifact_dir=self.tmp_model_dir
        )
        model.serialize_model(force_overwrite=True)
        assert os.path.exists(os.path.join(self.tmp_model_dir, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(self.tmp_model_dir, "config.json"))

    def test_serialize_as_onnx_warning(self, segmenter_pipeline):
        """
        Test serialize as_onnx=True gives warning.
        """

        with pytest.raises(
            NotImplementedError,
            match="HuggingFace Pipeline to onnx conversion is not supported.",
        ):
            model = HuggingFacePipelineModel(
                segmenter_pipeline, artifact_dir=self.tmp_model_dir
            )
            model.serialize_model(as_onnx=True, force_overwrite=True)

    def test_init(self, segmenter_pipeline):
        model = HuggingFacePipelineModel(
            segmenter_pipeline, artifact_dir=self.tmp_model_dir
        )
        assert model.algorithm == "ImageSegmentationPipeline"
        assert model.framework == "transformers"
        assert model.task == "image-segmentation"
        assert model.model_input_serializer.name == "cloudpickle"
        assert model.model_save_serializer.name == "huggingface"

    def test_prepare_verify(self, segmenter_pipeline):
        """
        Test serialize model input.
        """
        model = HuggingFacePipelineModel(
            segmenter_pipeline, artifact_dir=self.tmp_model_dir
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

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir, ignore_errors=True)
