State
*****

``ModelDeployer``
=================

The ``.get_model_deployment_state()`` method of the ``ModelDeployer`` class accepts a model deployment OCID and returns an ``enum`` state. This is a convenience method to obtain the model deployment state when the model deployment OCID is known. 

.. code-block:: python3

    from ads.model.deployment import ModelDeployer

    deployer = ModelDeployer()
    deployer.get_model_deployment_state(model_deployment_id="<MODEL_DEPLOYMENT_OCID>").name

.. parsed-literal::

    'ACTIVE'

``ModelDeployment``
===================

You can determine the state of the model deployment using the ``current_state.name`` attribute of a ``ModelDeployment`` object.  This returns a string with values like ‘ACTIVE’, ‘INACTIVE’, and ‘FAILED’.

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.get_model_deployment()``.

.. code-block:: python3

    deployment.current_state.name

