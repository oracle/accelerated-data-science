Access a Model Deployment
*************************

When a model is deployed the ``.deploy()`` method of the ``ModelDeployment`` class will return an updated ``ModelDeployment`` object. This object can be used to interact with the actual model deployment. However, if the model has already been deployed, it is possible to obtain a ``ModelDeployment`` object. Use the ``.from_id()`` method when the model deployment OCID is known.

The next code snippet creates a new ``ModelDeployment`` object that has access to the created model deployment.

.. code-block:: python3

    from ads.model.deployment import ModelDeployment

    existing_deployment = ModelDeployment.from_id(id="<MODEL_DEPLOYMENT_OCID>")

