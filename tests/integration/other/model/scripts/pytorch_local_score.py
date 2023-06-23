#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import sys
from functools import lru_cache
import torch
import numpy as np
import pandas as pd

model_name = "model.pt"


"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""

import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)


@lru_cache(maxsize=10)
def load_model(model_file_name=model_name):
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    model_dir = os.path.dirname(os.path.realpath(__file__))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    contents = os.listdir(model_dir)
    if model_file_name in contents:
        print(f"Start loading {model_file_name} from model directory {model_dir} ...")
        model_state_dict = torch.load(os.path.join(model_dir, model_file_name))
        print(f"loading {model_file_name} is complete.")
    else:
        raise Exception(
            f"{model_file_name} is not found in model directory {model_dir}"
        )

    # User would need to provide reference to the TheModelClass and
    # construct the the_model instance first before loading the parameters.
    the_model = SuperResolutionNet(upscale_factor=3)
    the_model.load_state_dict(model_state_dict)
    the_model.eval()
    print("Model is successfully loaded.")

    return the_model


def deserialize(data):
    """
        Deserialize json-serialized data to data in original type when sent to
    predict.

        Parameters
        ----------
        data: serialized input data.

        Returns
        -------
        data: deserialized input data.

    """
    json_data = data["data"]
    data_type = data["data_type"]

    if "numpy.ndarray" in data_type:
        return np.array(json_data)
    if "pandas.core.series.Series" in data_type:
        return pd.Series(json_data)
    if "pandas.core.frame.DataFrame" in data_type:
        return pd.read_json(json_data)
    if "torch.Tensor" in data_type:
        return torch.tensor(json_data)

    return json_data


def pre_inference(data):
    """
    Preprocess json-serialized data to feed into predict function.

    Parameters
    ----------
    data: Data format as expected by the predict API of the core estimator.

    Returns
    -------
    data: Data format after any processing.
    """
    data = deserialize(data)

    # Add further data preprocessing if needed
    from PIL import Image
    import io, base64

    img_bytes = io.BytesIO(base64.b64decode(data.encode("utf-8")))
    image = Image.open(img_bytes)

    import torchvision.transforms as transforms

    resize = transforms.Resize([224, 224])
    img = resize(image)

    img_ycbcr = img.convert("YCbCr")
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)

    img_y.unsqueeze_(0)
    return img_y


def post_inference(yhat):
    """
    Post-process the model results.

    Parameters
    ----------
    yhat: Data format after calling model.predict.

    Returns
    -------
    yhat: Data format after any processing.

    """
    if isinstance(yhat, torch.Tensor):
        return yhat.tolist()

    # Add further data postprocessing if needed
    return yhat


def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict.

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Pandas DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction': output from model.predict method}

    """

    img_y = pre_inference(data)

    with torch.no_grad():
        yhat = post_inference(model(img_y))
    return {"prediction": yhat}
