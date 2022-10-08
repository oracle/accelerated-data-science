Model Artifact
--------------

A model artifact is a collection of files used to create a model deployment. Some example files included in a model artifact are the serialized model, ``score.py``, and ``runtime.yaml``. You can store your model artifact in a local directory, in a ZIP or TAR format. Then use the ``.from_model_artifact()`` method to import the model artifact into the serialization model class. The ``.from_model_artifact()`` method takes the following parameters:

* ``artifact_dir`` (str): Artifact directory to store the files needed for deployment.
* ``auth`` (Dict, optional): Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``force_overwrite`` (bool, optional): Defaults to ``False``. If ``True``, it will overwrite existing files.
* ``model_file_name`` (str): The serialized model file name.
* ``properties`` (ModelProperties, optional): Defaults to ``None``. ``ModelProperties`` object required to save and deploy the model.
* ``uri`` (str): The path to the folder, ZIP, or TAR file that contains the model artifact. The model artifact must contain the serialized model, the ``score.py``, ``runtime.yaml`` and other files needed for deployment. The content of the URI is copied to the ``artifact_dir`` folder.

