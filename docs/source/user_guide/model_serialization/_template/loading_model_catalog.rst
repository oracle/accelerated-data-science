Model Catalog
-------------

To populate a serialization model object from a model stored in the model catalog, call the ``.from_model_catalog()`` method. This method uses the model OCID to download the model artifacts, write them to the ``artifact_dir``, and update the serialization model object. The ``.from_model_catalog()`` method takes the following parameters:

* ``artifact_dir`` (str): Artifact directory to store the files needed for deployment.
* ``auth`` (Dict, optional): Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``bucket_uri`` (str, optional). Defaults to ``None``. The OCI Object Storage URI where model artifacts will be copied to. The ``bucket_uri`` is only necessary for uploading large artifacts with size greater than 2GB. Example: ``oci://<bucket_name>@<namespace>/prefix/``.
* ``force_overwrite`` (bool, optional): Defaults to ``False``. If ``True``, it will overwrite existing files.
* ``model_id`` (str): The model OCID.
* ``model_file_name`` (str): The serialized model file name.
* ``overwrite_existing_artifact`` (bool, optional). Defaults to ``True``. Overwrite target bucket artifact if exists.
* ``properties`` (ModelProperties, optional): Defaults to None. Define the properties to save and deploy the model.
* ``**kwargs``:
    - ``compartment_id`` (str, optional): Compartment OCID. If not specified, the value will be taken from the environment variables.
    - ``timeout`` (int, optional): Defaults to 10 seconds. The connection timeout in seconds for the client.

