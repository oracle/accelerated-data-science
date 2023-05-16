You can use the ``.deploy()`` method to deploy a model. You must first save the model to the model catalog, and then deploy it.

.. admonition:: Deployment with Flex shape
    :class: note

    It is mandatory to provide ``ocpus`` and ``memory_in_gb`` values, when deploy with Flex instance shapes. They are set in ``deployment_ocpus`` and ``deployment_memory_in_gbs`` of the ``deploy()`` method.

The ``.deploy()`` method returns a ``ModelDeployment`` object.  Specify deployment attributes such as display name, instance type, number of instances,  maximum router bandwidth, and logging groups.  The API takes the following parameters:

See `API documentation <../../ads.model.html#id1>`__ for more details about the parameters.


.. admonition:: Tips
   :class: note

   * Providing ``deployment_access_log_id`` and ``deployment_predict_log_id`` helps in debugging your model inference setup.
   * Default Load Balancer configuration has bandwidth of 10 Mbps. `Refer service document to help you choose the right setup. <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_create.htm>`_ 
   * Check for supported instance shapes `here <https://docs.oracle.com/en-us/iaas/data-science/using/overview.htm#supported-shapes>`_ .
