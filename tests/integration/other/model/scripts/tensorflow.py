#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.framework.tensorflow_model import TensorFlowModel

import tensorflow as tf


class MyTFModel:
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, y_train = x_train[:1000], y_train[:1000]

    def training(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, epochs=1)

        return model


def tensorflow():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    myTFModel = MyTFModel().training()

    TEST_PREDICTION = myTFModel(x_test[:10])
    return {
        "framework": TensorFlowModel,
        "estimator": myTFModel,
        "artifact_dir": "./artifact_folder/tensorflow",
        "inference_conda_env": "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs_on_Python_3.7/1.0/generalml_p37_cpu_v1",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": x_test[:10],
        "y_true": y_test[:10],
        "onnx_data": x_test[:10].tolist(),
        "local_pred": TEST_PREDICTION,
        "score_py_path": None,
    }
