Activate
********

An inactive model deployment can be activated using a ``ModelDeployment`` object. See `Activate Model Deployments <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_manage.htm#model_dep_deactivate>`_ for more details.

If you have a ``ModelDeployment`` object, you can use the ``.activate()`` method to activate the model that is associated with that object. 

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.from_id()``.

.. code-block:: python3

   deployment.activate(wait_for_completion=True)

