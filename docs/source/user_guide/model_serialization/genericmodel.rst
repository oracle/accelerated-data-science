.. GenericModel:

GenericModel
************

Overview
========

The ``GenericModel`` class in ADS provides an efficient way to serialize almost any model class. This section demonstrates how to use the ``GenericModel`` class to prepare model artifacts, verify models, save models to the model catalog, deploy models, and perform predictions on model deployment endpoints.

The ``GenericModel`` class works with any unsupported model framework that has a ``.predict()`` method. For the most common model classes such as scikit-learn, XGBoost, LightGBM, TensorFlow, and PyTorch, and AutoML, we recommend that you use the ADS provided, framework-specific serializations models. For example, for a scikit-learn model, use SKLearnmodel. For other models, use the ``GenericModel`` class.

.. include:: _template/overview.rst

These simple steps take your trained model and will deploy it into production with just a few lines of code.

Initialize
==========

Instantiate a ``GenericModel()`` object by giving it any model object. It accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained model.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. ModelProperties object required to save and deploy the model.
* ``serialize: (bool, optional)``: Defaults to ``True``. If ``True`` the model will be serialized into a pickle file. If ``False``, you must set the ``model_file_name`` in the ``.prepare()`` method, serialize the model manually, and save it in the ``artifact_dir``. You will also need to update the ``score.py`` file to work with this model.

.. include:: _template/initialize.rst

Summary Status
==============

.. include:: _template/summary_status.rst

.. figure:: figures/summary_status.png
   :align: center

Model Deployment
================

Prepare
-------

The prepare step is performed by the ``.prepare()`` method. It creates several customized files used to run the model after it is deployed. These files include:

* ``input_schema.json``: A JSON file that defines the nature of the feature data. It includes information about the features. This includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``model.pkl``: This is the default filename of the serialized model. It can be changed with the ``model_file_name`` attribute. By default, the model is stored in a pickle file. The parameter ``as_onnx`` can be used to save it in the ONNX format.
* ``output_schema.json``: A JSON file that defines the nature of the dependent variable. This includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``runtime.yaml``: This file contains information that is needed to set up the runtime environment on the deployment server. It has information about which conda environment was used to train the model, and what environment should be used to deploy the model. The file also specifies what version of Python should be used.
* ``score.py``: This script contains the ``load_model()`` and ``predict()`` functions. The ``load_model()`` function understands the format the model file was saved in and loads it into memory. The ``predict()`` function is used to make inferences in a deployed model. There are also hooks that allow you to perform operations before and after inference. You are able to modify this script to fit your specific needs.

To create the model artifacts, use the ``.prepare()`` method. The ``.prepare()`` method includes parameters for storing model provenance information.

.. include:: _template/prepare.rst

Verify
------

.. include:: _template/verify.rst

* ``data (Union[dict, str, tuple, list, bytes])``. The data is used to test if the deployment works in the local environment.

In ``GenericModel``, data serialization is not supported. You can implement data serialization and deserialization in the ``score.py`` file.

Save
----

.. include:: _template/save.rst

Deploy
------

.. include:: _template/deploy.rst

Prepare, Save and Deploy Shortcut
---------------------------------
.. include:: _template/prepare_save_deploy.rst


Predict
-------

.. include:: _template/predict.rst

* ``data: Union[dict, str, tuple, list]``: JSON serializable data used for making inferences.

The ``.predict()`` and ``.verify()`` methods take the same data formats.

.. include:: _template/score.rst

Load
====

You can restore serialization models from model artifacts, from model deployments or from models in the model catalog. This section provides details on how to restore serialization models.

.. include:: _template/loading_model_artifact.rst

.. code-block:: python3

    from ads.model.generic_model import GenericModel

    model = GenericModel.from_model_artifact(
                    uri="/folder_to_your/artifact.zip",
                    model_file_name="model.pkl",
                    artifact_dir="/folder_store_artifact"
                )

.. include:: _template/loading_model_catalog.rst

.. code-block:: python3

    from ads.model.generic_model import GenericModel

    model = GenericModel.from_model_catalog(model_id="<model_id>",
                                            model_file_name="model.pkl",
                                            artifact_dir=tempfile.mkdtemp())

.. include:: _template/loading_model_deployment.rst

.. code-block:: python3

    from ads.model.generic_model import GenericModel

    model = GenericModel.from_model_deployment(
        model_deployment_id="<model_deployment_id>",
        model_file_name="model.pkl",
        artifact_dir=tempfile.mkdtemp())

Delete a Deployment
===================

.. include:: _template/delete_deployment.rst

Example
=======

By default, the ``GenericModel`` serializes to a pickle file. The following example, the user creates a model. In the prepare step, the user saves the model as a pickle file with the name ``toy_model.pkl``. Then the user verifies the model, saves it to the model catalog, deploys the model and makes a prediction. Finally, the user deletes the model deployment and then deletes the model.

.. code-block:: python3

    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    model = Toy()

    generic_model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    generic_model.summary_status()
    generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            model_file_name="toy_model.pkl",
            force_overwrite=True
         )
    generic_model.verify(2)
    model_id = generic_model.save()
    generic_model.deploy()
    generic_model.predict(2)
    generic_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

You can also use the shortcut ``.prepare_save_deploy()`` instead of calling ``.prepare()``, ``.save()`` and ``.deploy()`` seperately.

.. code-block:: python3

    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    estimator = Toy()

    model = GenericModel(estimator=estimator)
    model.summary_status()
    # If you are running the code inside a notebook session and using a service pack, `inference_conda_env` can be omitted.
    model.prepare_save_deploy(inference_conda_env="dataexpl_p37_cpu_v3")
    model.verify(2)
    model.predict(2)
    model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model.model_id)
