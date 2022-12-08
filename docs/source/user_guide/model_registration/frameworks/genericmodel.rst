.. GenericModel:

================
Other Frameworks
================

See `API Documentation <../../../ads.model.html#ads.model.generic_model.GenericModel>`__

Overview
========

The ``ads.model.generic_model.GenericModel`` class in ADS provides an efficient way to serialize almost any model class. This section demonstrates how to use the ``GenericModel`` class to prepare model artifacts, verify models, save models to the model catalog, deploy models, and perform predictions on model deployment endpoints.

The ``GenericModel`` class works with any unsupported model framework that has a ``.predict()`` method. For the most common model classes such as scikit-learn, XGBoost, LightGBM, TensorFlow, and PyTorch, and AutoML, we recommend that you use the ADS provided, framework-specific serializations models. For example, for a scikit-learn model, use SKLearnmodel. For other models, use the ``GenericModel`` class.

.. include:: ../_template/overview.rst

These simple steps take your trained model and will deploy it into production with just a few lines of code.

Prepare Model Artifact
======================

Instantiate a ``GenericModel()`` object by giving it any model object. It accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained model.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. ModelProperties object required to save and deploy the model.
* ``serialize: (bool, optional)``: Defaults to ``True``. If ``True`` the model will be serialized into a pickle file. If ``False``, you must set the ``model_file_name`` in the ``.prepare()`` method, serialize the model manually, and save it in the ``artifact_dir``. You will also need to update the ``score.py`` file to work with this model.

.. include:: ../_template/initialize.rst

Summary Status
==============

.. include:: ../_template/summary_status.rst

.. figure:: ../figures/summary_status.png
   :align: center



Example
=======

By default, the ``GenericModel`` serializes to a pickle file. The following example, the user creates a model. In the prepare step, the user saves the model as a pickle file with the name ``toy_model.pkl``. Then the user verifies the model, saves it to the model catalog, deploys the model and makes a prediction. Finally, the user deletes the model deployment and then deletes the model.

.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    model = Toy()

    generic_model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    generic_model.summary_status()

    generic_model.prepare(
            inference_conda_env="dbexp_p38_cpu_v1",
            model_file_name="toy_model.pkl",
            force_overwrite=True
         )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    generic_model.verify(2)

    # Register the model
    model_id = generic_model.save(display_name="Custom Model")

    # Deploy and create an endpoint for the XGBoost model
    generic_model.deploy(
        display_name="My Custom Model",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
    )

    print(f"Endpoint: {generic_model.model_deployment.url}")

    # Generate prediction by invoking the deployed endpoint
    generic_model.predict(2)

    # To delete the deployed endpoint uncomment the line below
    # generic_model.delete_deployment(wait_for_completion=True)

You can also use the shortcut ``.prepare_save_deploy()`` instead of calling ``.prepare()``, ``.save()`` and ``.deploy()`` seperately.

.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    estimator = Toy()

    model = GenericModel(estimator=estimator)
    model.summary_status()

    # If you are running the code inside a notebook session and using a service pack, `inference_conda_env` can be omitted.
    model.prepare_save_deploy(inference_conda_env="dbexp_p38_cpu_v1")
    model.verify(2)

    # Generate prediction by invoking the deployed endpoint
    model.predict(2)

    # To delete the deployed endpoint uncomment the line below
    # model.delete_deployment(wait_for_completion=True)
