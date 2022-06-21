Model Deployment
----------------

.. versionadded:: 2.6.2

To populate a serialization model object from a model deployment, call the ``.from_model_deployment()`` method. This method accepts a model deployment OCID. It downloads the model artifacts, writes them to the model artifact directory (``artifact_dir``), and updates the serialization model object. The ``.from_model_deployment()`` method takes the following parameters:

* ``artifact_dir`` (str): Artifact directory to store the files needed for deployment.
* ``auth`` (Dict, optional): Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()``. Supply the appropriate authentication signer and the ``**kwargs`` required to instantiate an ``IdentityClient`` object.
* ``force_overwrite`` (bool, optional): Defaults to ``False``. If ``True``, it will overwrite existing files in the artifact directory.
* ``model_deployment_id`` (str): The model deployment OCID.
* ``model_file_name`` (str): The serialized model file name.
* ``properties`` (ModelProperties, optional): Defaults to ``None``. Define the properties to save and deploy the model.
* ``**kwargs``:
    - ``compartment_id`` (str, optional): Compartment OCID. If not specified, the value will be taken from the environment variables.
    - ``timeout`` (int, optional): Defaults to 10 seconds. The connection timeout in seconds for the client.
