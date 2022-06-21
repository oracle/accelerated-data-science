Access a Model Deployment
*************************

When a model is deployed the ``.deploy()`` method of the ``ModelDeployer`` class will return a ``ModelDeployment`` object. This object can be used to interact with the actual model deployment. However, if the model has already been deployed, it is possible to obtain a ``ModelDeployment`` object. Use the ``.get_model_deployment()`` method when the model deployment OCID is known.

The next code snippet creates a new ``ModelDeployment`` object that has access to the created model deployment.

.. code-block:: python3

    from ads.model.deployment import ModelDeployer

    deployer = ModelDeployer()
    existing_deployment = deployer.get_model_deployment(model_deployment_id="<MODEL_DEPLOYMENT_OCID>")





