Attributes
**********

The ``ModelDeployment`` class has a number of attributes that are assigned by the system. They provide a mechanism to determine the state of the model deployment, the URI to make predictions, the model deployment OCID, etc.

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.get_model_deployment()``.

OCID
====

The ``.model_deployment_id`` of the ``ModelDeployment`` class specifies
the OCID of the model deployment.

.. code-block:: python3

    deployment.model_deployment_id

State
=====

You can determine the state of the model deployment using the ``.current_state`` enum attribute of a ``ModelDeployment`` object.  This returns an enum object and the string value can be determined with ``.current_state.name``. It will have values like ‘ACTIVE’, ‘INACTIVE’, and ‘FAILED’.

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.get_model_deployment()``.

.. code-block:: python3

    deployment.current_state.name


URL
===

The URL of the model deployment to use to make predictions using an HTTP request. The request is made to the URL given in the ``.url`` attribute of the ``ModelDeployment`` class. You can make HTTP requests to this endpoint to have the model make predictions, see the `Predict <predict.html>`__  section and `Invoking a Model Deployment <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-invoke.htm>`__ documentation for details.

.. code-block:: python3

    deployment.url

