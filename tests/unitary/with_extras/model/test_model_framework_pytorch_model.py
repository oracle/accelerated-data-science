#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - PyTorchModel
"""
import base64
import os
import shutil
from io import BytesIO

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uuid
from ads.model.framework.pytorch_model import PyTorchModel
from ads.model.serde.model_serializer import (
    PyTorchOnnxModelSaveSERDE,
    PytorchOnnxModelSerializer,
)

torch.manual_seed(1)

tmp_model_dir = "/tmp/model/"


def setup_module():
    os.makedirs(tmp_model_dir, exist_ok=True)


class LSTMTagger(nn.Module):
    """
    Create a simple LSTM Pytorch Model for Part-of-Speech Tagging
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class MyPyTorchModel:
    """
    Train a simple LSTM Pytorch Model for Part-of-Speech Tagging
    """

    training_data = [
        # Tags are: DET - determiner; NN - noun; V - verb
        # For example, the word "The" is a determiner
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ]

    word_to_ix = {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

    # keep them small
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def training_lstm(self):
        model = LSTMTagger(
            self.EMBEDDING_DIM,
            self.HIDDEN_DIM,
            len(self.word_to_ix),
            len(self.tag_to_ix),
        )
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(300):
            for sentence, tags in self.training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = self.prepare_sequence(sentence, self.word_to_ix)
                targets = self.prepare_sequence(tags, self.tag_to_ix)

                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()

        return model


class TestPyTorchModel:
    """Unittests for the PyTorchModel class."""

    myPyTorchModel = MyPyTorchModel().training_lstm()
    dummy_input = torch.tensor([1, 2, 1, 3], dtype=torch.long)
    X_sample = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    inference_conda_env = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
    inference_python_version = "3.6"

    def test_serialize_with_incorrect_model_file_name_onnx(self):
        """
        Test wrong model_file_name format.
        """
        test_pytorch_model = PyTorchModel(
            self.myPyTorchModel,
            tmp_model_dir,
        )
        with pytest.raises(AssertionError):
            test_pytorch_model._handle_model_file_name(
                as_onnx=True, model_file_name="model.xxx"
            )

    def test_serialize_using_pytorch_without_modelname(self):
        """
        Test serialize_model using pytorch without model_file_name
        """
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.model_file_name = test_pytorch_model._handle_model_file_name(
            as_onnx=False, model_file_name=None
        )
        test_pytorch_model.serialize_model(as_onnx=False)
        assert os.path.isfile(tmp_model_dir + "model.pt")

    def test_serialize_using_pytorch_with_modelname(self):
        """
        Test serialize_model using pytorch with correct model_file_name
        """
        test_pytorch_model = PyTorchModel(
            self.myPyTorchModel,
            tmp_model_dir,
        )
        test_pytorch_model.model_file_name = "test1.pt"
        test_pytorch_model.serialize_model(as_onnx=False)
        assert os.path.isfile(tmp_model_dir + "test1.pt")

    def test_serialize_using_onnx_without_modelname(self):
        """
        Test serialize_model using onnx without model_file_name
        """
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.model_file_name = test_pytorch_model._handle_model_file_name(
            as_onnx=True, model_file_name=None
        )
        test_pytorch_model.serialize_model(as_onnx=True, onnx_args=self.dummy_input)
        assert isinstance(
            test_pytorch_model.get_model_serializer(), PyTorchOnnxModelSaveSERDE
        )
        assert os.path.exists(os.path.join(tmp_model_dir, "model.onnx"))

    def test_serialize_using_onnx_with_modelname(self):
        """
        Test serialize_model using onnx with correct model_file_name
        """
        test_pytorch_model = PyTorchModel(
            self.myPyTorchModel,
            tmp_model_dir,
        )
        test_pytorch_model.model_file_name = f"model_{uuid.uuid4()}.onnx"
        test_pytorch_model.serialize_model(as_onnx=True, onnx_args=self.dummy_input)
        assert isinstance(
            test_pytorch_model.get_model_serializer(), PyTorchOnnxModelSaveSERDE
        )
        assert os.path.exists(
            os.path.join(tmp_model_dir, test_pytorch_model.model_file_name)
        )

    def test_to_onnx(self):
        """
        Test if PytorchOnnxModelSerializer.serialize generate onnx model result.
        """
        test_pytorch_model = PyTorchModel(
            self.myPyTorchModel,
            tmp_model_dir,
        )
        model_file_name = f"model_{uuid.uuid4()}.onnx"
        PytorchOnnxModelSerializer().serialize(
            estimator=test_pytorch_model.estimator,
            model_path=os.path.join(tmp_model_dir, model_file_name),
            onnx_args=self.dummy_input,
        )
        assert os.path.exists(os.path.join(tmp_model_dir, model_file_name))

    def test_to_onnx_reload(self):
        """
        Test if PytorchOnnxModelSerializer.serialize generate onnx model result.
        """
        test_pytorch_model = PyTorchModel(
            self.myPyTorchModel,
            tmp_model_dir,
        )
        model_file_name = f"model_{uuid.uuid4()}.onnx"
        PytorchOnnxModelSerializer().serialize(
            estimator=test_pytorch_model.estimator,
            model_path=os.path.join(tmp_model_dir, model_file_name),
            onnx_args=self.dummy_input,
        )
        assert (
            rt.InferenceSession(os.path.join(tmp_model_dir, model_file_name))
            is not None
        )

    def test_to_onnx_without_dummy_input(self):
        """
        Test if PytorchOnnxModelSerializer.serialize raise expected error
        """
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        model_file_name = f"model_{uuid.uuid4()}.onnx"
        with pytest.raises(ValueError):
            PytorchOnnxModelSerializer().serialize(
                estimator=test_pytorch_model.estimator,
                model_path=os.path.join(tmp_model_dir, model_file_name),
            )

    def test_to_onnx_without_path(self):
        """
        Test if PytorchOnnxModelSerializer.serialize raise expected error
        """
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        with pytest.raises(TypeError):
            PytorchOnnxModelSerializer().serialize(
                estimator=test_pytorch_model.estimator, onnx_args=self.dummy_input
            )

    @pytest.mark.parametrize(
        "test_data",
        [pd.Series([1, 2, 3]), [1, 2, 3]],
    )
    def test_get_data_serializer_with_convert_to_list(self, test_data):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        serialized_data = test_pytorch_model.get_data_serializer().serialize(test_data)
        assert serialized_data["data"] == [1, 2, 3]
        assert serialized_data["data_type"] == str(type(test_data))

    def test_get_data_serializer_helper_numpy(self):
        test_data = np.array([1, 2, 3])
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        serialized_data = test_pytorch_model.get_data_serializer().serialize(test_data)
        load_bytes = BytesIO(base64.b64decode(serialized_data["data"].encode("utf-8")))
        deserialized_data = np.load(load_bytes, allow_pickle=True)
        assert (deserialized_data == test_data).any()

    @pytest.mark.parametrize(
        "test_data",
        [
            pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}),
        ],
    )
    def test_get_data_serializer_with_pandasdf(self, test_data):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        serialized_data = test_pytorch_model.get_data_serializer().serialize(test_data)
        assert (
            serialized_data["data"]
            == '{"a":{"0":1,"1":2},"b":{"0":2,"1":3},"c":{"0":3,"1":4}}'
        )
        assert serialized_data["data_type"] == "<class 'pandas.core.frame.DataFrame'>"

    @pytest.mark.parametrize(
        "test_data",
        ["I have an apple", {"a": [1], "b": [2], "c": [3]}],
    )
    def test_get_data_serializer_with_no_change(self, test_data):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        serialized_data = test_pytorch_model.get_data_serializer().serialize(test_data)
        assert serialized_data["data"] == test_data

    def test_get_data_serializer_raise_error(self):
        class TestData:
            pass

        test_data = TestData()
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        with pytest.raises(TypeError):
            serialized_data = test_pytorch_model.get_data_serializer().serialize(
                test_data
            )

    def test_framework(self):
        """Test framework"""
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        assert test_pytorch_model.framework == "pytorch"

    def test_prepare_default(self):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
        )
        assert os.path.exists(tmp_model_dir + "model.pt")

    def test_prepare_onnx(self):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
            as_onnx=True,
            onnx_args=self.dummy_input,
        )
        assert os.path.exists(tmp_model_dir + "model.onnx")

    def test_prepare_onnx_with_X_sample(self):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
            as_onnx=True,
            X_sample=self.X_sample,
        )
        assert isinstance(test_pytorch_model.verify([1, 2, 3, 4]), dict)

    def test_prepare_onnx_without_input(self):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        with pytest.raises(ValueError):
            test_pytorch_model.prepare(
                inference_conda_env=self.inference_conda_env,
                inference_python_version=self.inference_python_version,
                force_overwrite=True,
                as_onnx=True,
            )

    def test_verify_onnx(self):
        """
        Test if PyTorchModel.verify in onnx serialization
        """
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
            as_onnx=True,
            onnx_args=self.dummy_input,
        )
        assert isinstance(test_pytorch_model.verify([1, 2, 3, 4]), dict)

    def test_save_as_torchscriptl(self):
        test_pytorch_model = PyTorchModel(self.myPyTorchModel, tmp_model_dir)
        test_pytorch_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
            use_torch_script=True,
        )
        assert isinstance(test_pytorch_model.verify(self.X_sample), dict)


def teardown_module():
    shutil.rmtree(tmp_model_dir)
