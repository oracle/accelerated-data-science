.. PyTorchModel:

PyTorchModel
************

See `API Documentation <../../../ads.model_framework.html#ads.model.framework.pytorch_model.PyTorchModel>`__

Overview
========

The ``ads.model.framework.pytorch_model.PyTorchModel`` class in ADS is designed to allow you to rapidly get a PyTorch model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: ../_template/overview.rst

The following steps take your trained ``PyTorch`` model and deploy it into production with a few lines of code.

**Create a PyTorch Model**

Load a `ResNet18 <https://arxiv.org/pdf/1512.03385.pdf>`_ model and put it into evaluation mode.

.. code-block:: python3

    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

Prepare Model Artifact
======================

Save as TorchScript
-------------------
.. versionadded:: 2.6.9

Serializing model in TorchScript program by setting `use_torch_script` to `True`, you can load the model and run inference without defining the model class.

.. code-block:: python3

    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.pytorch_model import PyTorchModel

    import tempfile

    # Prepare the model
    artifact_dir = "pytorch_model_artifact"
    pytorch_model = PyTorchModel(model, artifact_dir=artifact_dir)
    pytorch_model.prepare(
        inference_conda_env="pytorch110_p38_cpu_v1",
        training_conda_env="pytorch110_p38_cpu_v1",
        use_case_type=UseCaseType.IMAGE_CLASSIFICATION,
        force_overwrite=True,
        use_torch_script=True,
    )
    # You don't need to modify the score.py generated. The model can be loaded without defining the model class.
    # More info here - https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

Save state_dict
---------------
.. code-block:: python3

    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.pytorch_model import PyTorchModel

    import tempfile

    # Prepare the model
    artifact_dir = "pytorch_model_artifact"
    pytorch_model = PyTorchModel(model, artifact_dir=artifact_dir)
    pytorch_model.prepare(
        inference_conda_env="pytorch110_p38_cpu_v1",
        training_conda_env="pytorch110_p38_cpu_v1",
        use_case_type=UseCaseType.IMAGE_CLASSIFICATION,
        force_overwrite=True,
    )

    # The score.py generated requires you to create the class instance of the Model before the weights are loaded.
    # More info here - https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended

Open ``pytorch_model_artifact/score.py`` and edit the code to instantiate the model class. The edits are highlighted -

.. code-block:: python3
    :emphasize-lines: 13,14

    import os
    import sys
    from functools import lru_cache
    import torch
    import json
    from typing import Dict, List
    import numpy as np
    import pandas as pd
    from io import BytesIO
    import base64
    import logging

    import torchvision
    the_model = torchvision.models.resnet18()

    model_name = 'model.pt'



    """
    Inference script. This script is used for prediction by scoring server when schema is known.
    """

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
            print(f'Start loading {model_file_name} from model directory {model_dir} ...')
            model_state_dict = torch.load(os.path.join(model_dir, model_file_name))
            print(f"loading {model_file_name} is complete.")
        else:
            raise Exception(f'{model_file_name} is not found in model directory {model_dir}')

        # User would need to provide reference to the TheModelClass and
        # construct the the_model instance first before loading the parameters.
        # the_model = TheModelClass(*args, **kwargs)
        try:
            the_model.load_state_dict(model_state_dict)
        except NameError as e:
            raise NotImplementedError("TheModelClass instance must be constructed before loading the parameters. Please modify the load_model() function in score.py." )
        except Exception as e:
            raise e

        the_model.eval()
        print("Model is successfully loaded.")

        return the_model


Instantiate a ``PyTorchModel()`` object with a PyTorch model. Each instance accepts the following parameters:

* ``artifact_dir: str``. Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: Callable``. Any model object generated by the PyTorch framework.
* ``properties: (ModelProperties, optional)``. Defaults to ``None``. The ``ModelProperties`` object required to save and deploy model.

.. include:: ../_template/initialize.rst


Verify Changes to Score.py
==========================

Download and load an image for prediction

.. code-block:: python3

    # Download an image
    import urllib.request
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # Preprocess the image and convert to torch.Tensor
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

Verify ``score.py`` changes by running inference locally

.. code-block:: python3

    >>> prediction = pytorch_model.verify(input_batch)["prediction"]
    >>> import numpy as np
    >>> np.argmax(prediction)
    258

Summary Status
==============

.. include:: ../_template/summary_status.rst

.. figure:: ../figures/summary_status.png
   :align: center

Register Model
==============

.. code-block:: python3

    >>> # Register the model
    >>> model_id = pytorch_model.save()

    Start loading model.pt from model directory /tmp/tmpf11gnx9c ...
    loading model.pt is complete.
    Model is successfully loaded.
    ['.score.py.swp', 'score.py', 'model.pt', 'runtime.yaml']


    'ocid1.datasciencemodel.oc1.xxx.xxxxx'

Deploy and Generate Endpoint
============================

.. code-block:: python3

    >>> # Deploy and create an endpoint for the PyTorch model
    >>> pytorch_model.deploy(
            display_name="PyTorch Model For Classification",
            deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
            deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
            deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        )


    >>> print(f"Endpoint: {pytorch_model.model_deployment.url}")

    https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx

Run Prediction against Endpoint
===============================

.. code-block:: python3

    # Download an image
    import urllib.request
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # Preprocess the image and convert to torch.Tensor
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


    # Generate prediction by invoking the deployed endpoint
    prediction = pytorch_model.predict(input_batch)['prediction']
    print(np.argmax(prediction))

.. parsed-literal::

    258

Predict with Image
------------------
.. versionadded:: 2.6.7

Predict Image by passing a uri, which can be http(s), local path, or other URLs
(e.g. starting with “oci://”, “s3://”, and “gcs://”), of the image or a PIL.Image.Image object
using the ``image`` argument in ``predict()`` to predict a single image.
The image will be converted to a tensor and then serialized so it can be passed to the endpoint.
You can catch the tensor in ``score.py`` to perform further transformation.

.. note::
    The payload size limit is 10 MB. Read more about invoking a model deployment `here <https://docs.oracle.com/iaas/data-science/using/model-dep-invoke.htm#model_dep_invoke>`_.

Given the size limitation, the example below is with resized image. To pass an image and invoke prediction, additional code inside score.py is required to preprocess the data. Open ``pytorch_model_artifact/score.py`` and update the ``pre_inference()`` method. The edits are highlighted:

.. code-block:: python3
    :emphasize-lines: 15-26

    def pre_inference(data, input_schema_path):
        """
        Preprocess json-serialized data to feed into predict function.

        Parameters
        ----------
        data: Data format as expected by the predict API of the core estimator.
        input_schema_path: path of input schema.

        Returns
        -------
        data: Data format after any processing.
        """
        data = deserialize(data, input_schema_path)
        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        input_tensor = preprocess(data)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

Save ``score.py`` and verify prediction works:

.. code-block:: python3

    >>> uri = ("https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/model_deploy_examples/images/dog_resized.jpg")
    >>> prediction = pytorch_model.verify(image=uri)["prediction"]
    >>> import numpy as np
    >>> np.argmax(prediction)
    258

Re-deploy model with updated ``score.py``:

.. code-block:: python3

    pytorch_model.deploy(
        display_name="PyTorch Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxxx",
    )

Run prediction with the image provided:

.. code-block:: python3

    uri = ("https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/model_deploy_examples/images/dog_resized.jpg")

    # Generate prediction by invoking the deployed endpoint
    prediction = pytorch_model.predict(image=uri)['prediction']

Run Prediction with oci raw-request command
===========================================

Model deployment endpoints can be invoked with the OCI-CLI. This example invokes a model deployment with the CLI with a ``torch.Tensor`` payload:

`torch.Tensor` payload example
------------------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction and save it to file 'data-payload'
    >>> from io import BytesIO
    >>> import base64

    >>> buffer = BytesIO()
    >>> torch.save(input_batch, buffer)
    >>> data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    >>> with open('data-payload', 'w') as f:
    >>>     f.write('{"data": "' + data + '", "data_type": "torch.Tensor"}')

File ``data-payload`` will have this information:

.. code-block:: bash

    {"data": "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQ ........................
    .......................................................................
    ...AAAAEAAABQSwUGAAAAAAMAAwC3AAAA0jEJAAAA", "data_type": "torch.Tensor"}

Use file ``data-payload`` with data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body file://data-payload

Expected output of raw-request command
--------------------------------------

.. code-block:: bash

    {
      "data": {
        "prediction": [
          [
            0.0159152802079916,
            -1.5496551990509033,
            .......
            2.5116958618164062
          ]
        ]
      },
      "headers": {
        "Connection": "keep-alive",
        "Content-Length": "19398",
        "Content-Type": "application/json",
        "Date": "Thu, 08 Dec 2022 18:28:41 GMT",
        "X-Content-Type-Options": "nosniff",
        "opc-request-id": "BD80D931A6EA4C718636ECE00730B255/86111E71C1B33C24988C59C27F15ECDE/E94BBB27AC3F48CB68F41135073FF46B",
        "server": "uvicorn"
      },
      "status": "200 OK"
    }

Copy prediction output into `argmax` to retrieve result of the image prediction:

.. code-block:: python3

    >>> print(np.argmax([[0.0159152802079916,
    >>>                   -1.5496551990509033,
    >>>                   .......
    >>>                   2.5116958618164062]]))
    258

Example
=======

.. code-block:: python3

    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.pytorch_model import PyTorchModel

    import numpy as np
    from PIL import Image

    import tempfile
    import torchvision
    from torchvision import transforms

    import urllib

    # Load a pretrained PyTorch Model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()


    # Prepare Model Artifact for PyTorch Model
    artifact_dir = tempfile.mkdtemp()
    pytorch_model = PyTorchModel(model, artifact_dir=artifact_dir)
    pytorch_model.prepare(
        inference_conda_env="pytorch110_p38_cpu_v1",
        training_conda_env="pytorch110_p38_cpu_v1",
        use_case_type=UseCaseType.IMAGE_CLASSIFICATION,
        force_overwrite=True,
        use_torch_script=True
    )


    # Download an image for running inference
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    # Load image
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    prediction = pytorch_model.verify(input_batch)["prediction"]
    print(np.argmax(prediction))

    # Register the model
    model_id = pytorch_model.save(display_name="PyTorch Model")

    # Deploy and create an endpoint for the PyTorch model
    pytorch_model.deploy(
        display_name="PyTorch Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxxx",
    )

    # Generate prediction by invoking the deployed endpoint
    prediction = pytorch_model.predict(input_batch)["prediction"]

    print(np.argmax(prediction))

    # To delete the deployed endpoint uncomment the line below
    # pytorch_model.delete_deployment(wait_for_completion=True)


