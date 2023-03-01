Update
******

The ``.update()`` method of the ``ModelDeployment`` class is used to make changes to a deployed model. Check out the `Editing Model Deployments <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_manage.htm>`__ for a
list of what properties can be updated.

A common use case is to change the underlying model that is deployed. In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.from_id()``.

.. code-block:: python3

    deployment.runtime.with_model_uri("<NEW_MODEL_OCID>")
    deployment.update()

Or, you could update the instance shape with:

.. code-block:: python3

    deployment.infrastructure.with_shape_name("VM.Standard.E4.Flex").with_shape_config_details(ocpus=2, memory_in_gbs=32)
    deployment.update()

