#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.framework.pytorch_model import PyTorchModel

import os
import PIL

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

tmp_model_dir1 = "/tmp/model_pytorch"
tmp_model_dir2 = "/tmp/model_pytorch_onnx"
DIFF = 0.0001

inference_conda_env = "oci://service-conda-packs@ociodscdev/service_pack/cpu/Computer Vision for CPU on Python 3.7/1.0/computervision_p37_cpu_v1"
inference_python_version = "3.7"


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


class MyPyTorchModel:
    torch_model = SuperResolutionNet(upscale_factor=3)

    def training(self):
        model_url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"
        batch_size = 1  # just a random number

        # Initialize model with the pretrained weights
        map_location = lambda storage, loc: storage
        if torch.cuda.is_available():
            map_location = None
        self.torch_model.load_state_dict(
            model_zoo.load_url(model_url, map_location=map_location)
        )

        # set the model to inference mode
        self.torch_model.eval()
        return self.torch_model

    def load_test_image_to_arr(self):
        # test image set up
        test_image = PIL.Image.open(
            os.path.join(
                f"{os.path.dirname(os.path.abspath(__file__))}/../image_files/dog.jpeg",
            )
        )

        import torchvision.transforms as transforms

        resize = transforms.Resize([224, 224])
        img = resize(test_image)

        img_ycbcr = img.convert("YCbCr")
        img_y, img_cb, img_cr = img_ycbcr.split()
        to_tensor = transforms.ToTensor()
        img_y = to_tensor(img_y)
        test_image_arr = img_y.unsqueeze_(0).detach().numpy().tolist()
        return img_y, test_image_arr

    def load_test_image_to_str(self):
        # test image set up
        import os, base64

        path = os.path.join(
            f"{os.path.dirname(os.path.abspath(__file__))}/../image_files/dog.jpeg",
        )

        img_bytes = open(os.path.join(path), "rb").read()
        img_inputs = base64.b64encode(img_bytes).decode("utf8")
        return img_inputs


def pytorch():
    my_torch_model = MyPyTorchModel().training()
    test_img_y, test_image_arr = MyPyTorchModel().load_test_image_to_arr()
    img_str = MyPyTorchModel().load_test_image_to_str()
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    TEST_PREDICTION = my_torch_model(test_img_y)
    return {
        "framework": PyTorchModel,
        "estimator": my_torch_model,
        "artifact_dir": "./artifact_folder/pytorch",
        "inference_conda_env": inference_conda_env,
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": img_str,
        "y_true": TEST_PREDICTION,
        "onnx_data": test_image_arr,
        "prepare_args": {"onnx_args": dummy_input},
        "local_pred": TEST_PREDICTION,
        "score_py_path": "scripts/pytorch_local_score.py",
    }
