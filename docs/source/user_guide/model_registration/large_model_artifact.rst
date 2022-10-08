Large Model Artifacts
*********************

.. versionadded:: 2.6.4

Large models are models with artifacts between 2 and 6 GB. You must first upload large models to an Object Storage bucket, and then transfer them to a model catalog. Follow a similar process to download a model artifact from the model catalog. First download large models from the model catalog to an Object Storage bucket, and then transfer them to local storage. For model artifacts that are less than 2 GB, you can use the same approach, or download them directly to local storage.

ADS :doc:`framework specific wrapper <quick_start>` classes save large models using a process almost identical to model artifacts that are less than 2GB. An Object Storage bucket is required with Data Science service access granted to that bucket.

If you don't have an Object Storage bucket, create one using the OCI SDK or the Console. Create an `Object Storage bucket <https://docs.oracle.com/iaas/Content/Object/home.htm>`_. Make a note of the namespace, compartment, and bucket name. Configure the following policies to allow the Data Science service to read and write the model artifact to the Object Storage bucket in your tenancy. An administrator must configure these policies in `IAM <https://docs.oracle.com/iaas/Content/Identity/home1.htm>`_ in the Console.

.. parsed-literal::

        Allow service datascience to manage object-family in compartment <compartment> where ALL {target.bucket.name='<bucket_name>'}

        Allow service objectstorage to manage object-family in compartment <compartment> where ALL {target.bucket.name='<bucket_name>'}

See `API documentation <../../ads.model.html#id10>`__ for more details.

Saving
======

We recommend that you work with model artifacts using the :doc:`framework specific wrapper <quick_start>` classes in ADS. After you prepare and verify the model, the model is ready to be stored in the model catalog. The standard method to do this is to use the ``.save()`` method. If the ``bucket_uri`` parameter is present, then the large model artifact is supported.

The URI syntax for the ``bucket_uri`` is:

``oci://<bucket_name>@<namespace>/<path>/``

The following saves the :doc:`framework specific wrapper <quick_start>` object, ``model``, to the model catalog and returns the OCID from the model catalog:

.. code-block:: python3

   model_catalog_id = model.save(
        display_name='Model With Large Artifact',
        bucket_uri=<provide bucket url>,
        overwrite_existing_artifact = True,
        remove_existing_artifact = True,
    )

Loading
=======

We recommend that you transfer a model artifact from the model catalog to your notebook session using the :doc:`framework specific wrapper <quick_start>` classes in ADS. The ``.from_model_catalog()`` method takes the model catalog OCID and some file parameters. If the ``bucket_uri`` parameter is present, then a large model artifact is used.

The following example downloads a model from the model catalog using the large model artifact approach. The ``bucket_uri`` has the following syntax:

``oci://<bucket_name>@<namespace>/<path>/``

.. code-block:: python3

    large_model = model.from_model_catalog(
        model_id=model_catalog_id,
        model_file_name="model.pkl",
        artifact_dir="./artifact/",
        bucket_uri=<provide bucket url> ,
        force_overwrite=True,
        remove_existing_artifact=True,
    )
