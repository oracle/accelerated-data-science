=======================================
Dowloading Models from OCI Data Science
=======================================

Download Registered Model
=========================

Download and recreate :doc:`framework specific wrapper <framework_specific_instruction>` objects using the ``ocid`` value of your model. 

The downloaded artifact can be used for running inferece in local environment. You can update the artifact files to change your ``score.py`` or model and then register as a new model. See :doc:`here <model_file_customization>` to learn how to change ``score.py``

Here is an example for loading back a LightGBM model that was previously registered.

.. code-block:: python3

    from ads.model.framework.lightgbm_model import LightGBMModel

    lgbm_model = LightGBMModel.from_model_catalog(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
    )

.. parsed-literal:: 

    Model is successfully loaded.

See `API doc <../../ads.model.html#id3>`__ for more infomation.

Download Deployed Model
=======================

Download and recreate :doc:`framework specific wrapper <framework_specific_instruction>` objects using the ``ocid`` value of your OCI Model Deployment instance. 

The downloaded artifact can be used for running inferece in local environment. You can update the artifact files to change your ``score.py`` or model and then register as a new model. See :doc:`here <model_file_customization>` to learn how to change ``score.py``

Here is an example for loading back a LightGBM model that was previously deployed.

.. code-block:: python3

    from ads.model.framework.pytorch_model import PyTorchModel

    pytorchmodel = PyTorchModel.from_model_deployment(
        "ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx",
        model_file_name="model.pt",
        artifact_dir="pytorch-download-test",
    )

    print(pytorchmodel.model_deployment.url)

.. parsed-literal:: 

    Start loading model.pt from model directory /home/datascience/pytorch-download-test ...
    loading model.pt is complete.
    Model is successfully loaded.

    https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxx

See `API doc <../../ads.model.html#id4>`__ for more infomation.


