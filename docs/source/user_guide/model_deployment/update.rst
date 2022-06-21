Update
******

The ``.update()`` method of the ``ModelDeployment`` class is used to make changes to a deployed model. This method accepts the same parameters as the ``.deploy()`` method. Check out the `Editing Model Deployments <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_manage.htm>`__ for a
list of what properties can be updated.

A common use case is to change the underlying model that is deployed. In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.get_model_deployment()``.

.. code-block:: python3

    deployment.update(model_id="<NEW_MODEL_OCID>")

Or, you could update the instance shape with:

.. code-block:: python3

    deployment.update(
        model_deployment_properties.with_instance_configuration(
            dict(instance_shape="VM.Standard2.1")
        )
    )

