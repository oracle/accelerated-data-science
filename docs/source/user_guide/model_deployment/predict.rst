Predict
*******

Predictions can be made by calling the HTTP endpoint associated with the model deployment. The ``ModelDeployment`` object ``url`` attribute specifies the endpoint. You could also use the ``ModelDeployment`` object with the ``.predict()`` method. The format of the data that is passed to the HTTP endpoint depends on the setup of the model artifact. The default setup is to pass in a Python dictionary that has been converted to a JSON data structure. The first level defines the feature names. The second level uses an identifier for the observation (for example, row in the dataframe), and the value associated with it. Assuming the model has features F1, F2, F3, F4, and F5, then the observations are identified by the values 0, 1, and 2 and the data would look like this:

===== == == == == ==
Index F1 F2 F3 F4 F5
===== == == == == ==
0     11 12 13 14 15
1     21 22 23 24 25
2     31 32 33 34 35
===== == == == == ==

The Python dictionary representation would be:

.. code-block:: python3

   test = { 
      'F1': { 0: 11, 1: 21, 2: 31},
      'F2': { 0: 12, 1: 22, 2: 32},
      'F3': { 0: 13, 1: 23, 2: 33},
      'F4': { 0: 14, 1: 24, 2: 34},
      'F5': { 0: 15, 1: 25, 2: 35}
   }


You can use the ``ModelDeployment`` object to call the HTTP endpoint. The returned
result is the predictions for the three observations.

.. code-block:: python3

    deployment.predict(test)

.. parsed-literal::

    {'prediction': [0, 2, 0]}


Model Deploy now supports binary payloads. You no longer need to convert binary images to Base64 encoded strings when making inferences.

Example
=======

The following example shows how to use `predict()` with image bytes:
The `score.py` file does not provide default deserialization for bytes input. You need to provide your own implementations. 
The model used in this example has its raw training data `normalized <https://pytorch.org/hub/pytorch_vision_resnet/>`_. The next cell reproduces these transformations. The original image is 256x384 pixels and the training data is 224x224. Therefore, the image is resized and cropped. The color variation in the image is also adjusted to match the training data. The image is converted to a `Tensor` object. This object is a four-dimensional tensor and the first dimension has only a single level. This dimension is removed using the `.unsqueeze()` method.

Load data

.. code-block:: python3

   from PIL import Image
   im = Image.open('<image_path>')
   im.convert("RGB").save("<image_path>")

   with open('<image_path>', 'rb') as f:
      byte_im = f.read()

Example model

.. code-block:: python3

   #  load the pre-trained model.
   model = resnet18(pretrained=True)
   # set the model to inference mode
   _ = model.eval()

Model framework serialization

.. code-block:: python3

   artifact_dir = "<directory>"
   pytorch_model = PyTorchModel(estimator=model, artifact_dir=artifact_dir)
   conda_env = 'computervision_p37_cpu_v1'

   # Create a sample of the y values.
   y_sample = [0] * len(prediction_not_normalized)
   y_sample[prediction_normalized.index(max_value)] = 1

   pytorch_model.prepare(
      inference_conda_env=conda_env,
      training_conda_env=conda_env,
      use_case_type=UseCaseType.IMAGE_CLASSIFICATION,
      X_sample=image_tensor,
      y_sample=y_sample,
      training_id=None,
      force_overwrite=True
   )
   pytorch_model.verify(byte_im)['prediction'][0][:10]
   model_id = pytorch_model.save(display_name='Test PyTorchModel model Bytes Input', timeout=600)

   deploy = pytorch_model.deploy(display_name='Test PyTorchModel deployment')
   pytorch_model.predict(byte_im)['prediction'][0][:10]

   pytorch_model.delete_deployment(wait_for_completion=True)
   ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


The change needed in `score.py`:

.. code-block:: python3

   def deserialize(data):
      if isinstance(data, bytes):
         return data

         ...
   
   
   def pre_inference(data):
      data = deserialize(data)

      import base64
      import io
      import torchvision.transforms as transforms
      
      from PIL import Image
      img_bytes = io.BytesIO(data)
      image = Image.open(img_bytes)

      # preprocess the data to make it accepted by the model
      preprocess = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
               mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225]
         ),
      ])
      input_tensor = preprocess(image)
      input_batch = input_tensor.unsqueeze(0)
      
      return input_batch

   def post_inference(yhat):
      if isinstance(yhat, torch.Tensor):
         from torch.nn import Softmax
         softmax = Softmax(dim=1)
         return softmax(yhat).tolist()
      
      return yhat