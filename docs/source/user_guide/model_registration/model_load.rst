Load Registered Model
=====================

Load and recreate :doc:`framework specific wrapper <framework_specific_instruction>` objects using the ``ocid`` value of your model.

The loaded artifact can be used for running inference in local environment. You can update the artifact files to change your ``score.py`` or model and then register as a new model. See :doc:`here <model_file_customization>` to learn how to change ``score.py``

Here is an example for loading back a ``LightGBM`` model that was previously registered. See `API documentation <../../ads.model.html#id4>`__ for more details.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_catalog(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
    )

.. include:: ./_template/print.rst

.. include:: ./_template/summary_status.rst

.. figure:: ./figures/summary_status.png
   :align: center

.. versionadded:: 2.6.9

Alternatively the ``.from_id()`` method can be used to load a model. In future releases, the ``.from_model_catalog()`` method will be deprecated and replaced with the ``from_id()``. See `API documentation <../../ads.model.html#id2>`__ for more details.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_id(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        bucket_uri=<oci://<bucket_name>@<namespace>/prefix/>,
        force_overwrite=True,
        remove_existing_artifact=True,
    )


Load Deployed Model
===================

Load and recreate :doc:`framework specific wrapper <framework_specific_instruction>` objects using the ``ocid`` value of your OCI Model Deployment instance.

The loaded artifact can be used for running inference in local environment. You can update the artifact files to change your ``score.py`` or model and then register as a new model. See :doc:`here <model_file_customization>` to learn how to change ``score.py``

Here is an example for loading back a LightGBM model that was previously deployed. See `API doc <../../ads.model.html#id5>`__ for more infomation.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_deployment(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
    )

.. include:: ./_template/print.rst


.. versionadded:: 2.6.9

Alternatively the ``.from_id()`` method can be used to load a model from the Model Deployment. In future releases, the ``.from_model_deployment()`` method will be deprecated and replaced with the ``from_id()``. See `API documentation <../../ads.model.html#id2>`__ for more details.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_id(
        "ocid1.datasciencemodeldeployment.oc1.xxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        bucket_uri=<oci://<bucket_name>@<namespace>/prefix/>,
        force_overwrite=True,
        remove_existing_artifact=True,
    )

Load Model From Object Storage
==============================

Load and recreate :doc:`framework specific wrapper <framework_specific_instruction>` objects from the existing model artifact archive.

The loaded artifact can be used for running inference in local environment. You can update the artifact files to change your ``score.py`` or model and then register as a new model. See :doc:`here <model_file_customization>` to learn how to change ``score.py``

Here is an example for loading back a LightGBM model that was previously saved to the Object Storage. See `API doc <../../ads.model.html#id3>`__ for more infomation.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_artifact(
        <oci://<bucket_name>@<namespace>/prefix/lgbm_model_artifact.zip>,
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        force_overwrite=True
    )

A model loaded from an artifact archive can be registered and deployed.

.. include:: ./_template/print.rst


Large Model Artifacts
=====================

.. versionadded:: 2.6.4

Large models are models with artifacts between 2 and 6 GB. You must first download large models from the model catalog to an Object Storage bucket, and then transfer them to local storage. For model artifacts that are less than 2 GB, you can use the same approach, or download them directly to local storage. An Object Storage bucket is required with Data Science service access granted to that bucket.

If you don't have an Object Storage bucket, create one using the OCI SDK or the Console. Create an `Object Storage bucket <https://docs.oracle.com/iaas/Content/Object/home.htm>`_. Make a note of the namespace, compartment, and bucket name. Configure the following policies to allow the Data Science service to read and write the model artifact to the Object Storage bucket in your tenancy. An administrator must configure these policies in `IAM <https://docs.oracle.com/iaas/Content/Identity/home1.htm>`_ in the Console.

.. parsed-literal::

        Allow service datascience to manage object-family in compartment <compartment> where ALL {target.bucket.name='<bucket_name>'}

        Allow service objectstorage to manage object-family in compartment <compartment> where ALL {target.bucket.name='<bucket_name>'}

The following example loads a model using the large model artifact approach. The ``bucket_uri`` has the following syntax: ``oci://<bucket_name>@<namespace>/<path>/`` See `API documentation <../../ads.model.html#id4>`__ for more details.


.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_catalog(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        bucket_uri=<oci://<bucket_name>@<namespace>/prefix/>,
        force_overwrite=True,
        remove_existing_artifact=True,
    )


Here is an example for loading back a ``LightGBM`` model with large artifact from Model Deployment. See `API doc <../../ads.model.html#id5>`__ for more infomation.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_deployment(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        bucket_uri=<oci://<bucket_name>@<namespace>/prefix/>,
        force_overwrite=True,
        remove_existing_artifact=True,
    )


.. versionadded:: 2.6.9

Alternatively the ``.from_id()`` method can be used to load registered or deployed model. In future releases, the ``.from_model_catalog()`` and ``.from_model_deployment()`` methods will be deprecated and replaced with the ``from_id()``. See `API documentation <../../ads.model.html#id2>`__ for more details.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_id(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        bucket_uri=<oci://<bucket_name>@<namespace>/prefix/>,
        force_overwrite=True,
        remove_existing_artifact=True,
    )