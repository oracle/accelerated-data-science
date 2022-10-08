Deploy
******

The ``.deploy()`` method of the ``ModelDeployer`` class is used to create a model deployment.  It has the following parameters:

* ``max_wait_time``: The timeout limit, in seconds, for the deployment process to wait until it is active. Defaults to 1200 seconds.
* ``poll_interval``: The interval between checks of the deployment status in seconds. Defaults to 30 seconds.
* ``wait_for_completion``: Blocked process until the deployment has been completed. Defaults to ``True``.

There are two ways to use the ``.deploy()`` method. You can create a ``ModelDeploymentProperties`` object and pass that in, or you can define the model deployment properties using the ``.deploy()`` method.

With ``ModelDeploymentProperties``
==================================

After a ``ModelDeploymentProperties`` object is created, then you use ``model_deployment_properties`` to deploy a model as in this example:

.. code-block:: python3

    from ads.model.deployment import ModelDeployer, ModelDeploymentProperties

    model_deployment_properties = ModelDeploymentProperties(
       "<oci://your_bucket@your_namespace/path/to/dir>"
    ).with_prop(
        'display_name', "Model Deployment Demo using ADS"
    ).with_prop(
        "project_id", "<PROJECT_OCID>"
    ).with_prop(
        "compartment_id", "<COMPARTMENT_OCID>"
    ).with_logging_configuration(
        "<ACCESS_LOG_GROUP_OCID>", "<ACCESS_LOG_OCID>", "<PREDICT_LOG_GROUP_OCID>", "<PREDICT_LOG_OCID>"
    ).with_instance_configuration(
        config={
            "INSTANCE_SHAPE":"VM.Standard.E4.Flex",
            "INSTANCE_COUNT":"1",
            "bandwidth_mbps":10,
            "memory_in_gbs":10,
            "ocpus":1
        }
    )
    deployer = ModelDeployer()
    deployment = deployer.deploy(model_deployment_properties)


Without ``ModelDeploymentProperties``
=====================================

Depending on your use case, it might be more convenient to skip the creation of a ``ModelDeploymentProperties`` object and create the model deployment directly using the ``.deploy()`` method. You can do this by passing the using keyword arguments instead of ``ModelDeploymentProperties``. You specify the model deployment properties as parameters in the ``.deploy()`` method.

You define the model deployment properties using the following parameters:

* ``access_log_group_id``: Log group OCID for the access logs. Required when ``access_log_id`` is specified.
* ``access_log_id``: Custom logger OCID for the access logs. Required when ``access_log_group_id`` is specified.
* ``bandwidth_mbps``: The bandwidth limit on the load balancer in Mbps. Optional.
* ``compartment_id``: Compartment OCID that the model deployment belongs to.
* ``defined_tags``: A dictionary of defined tags to be attached to the model deployment. Optional.
* ``description``: A description of the model deployment. Optional.
* ``display_name``: A name that identifies the model deployment in the Console.
* ``freeform_tags``: A dictionary of freeform tags to be attached to the model deployment. Optional.
* ``instance_count``: The number of instances to deploy.
* ``instance_shape``: The instance compute shape to use. For example, “VM.Standard2.1”.
* ``memory_in_gbs``:  The size of the memory of the model deployment instance in GBs. Applicable for the flexible shapes, for example, “VM.Standard.E4.Flex”.
* ``model_id``: Model OCID that is used in the model deployment.
* ``ocpus``: The ocpus count of the model deployment instance. Applicable for the flexible shapes, for example, “VM.Standard.E4.Flex”.
* ``predict_log_group_id``: Log group OCID for the predict logs. Required when ``predict_log_id`` is specified.
* ``predict_log_id``: Custom logger OCID for the predict logs. Required when ``predict_log_group_id`` is specified.
* ``project_id``: Project OCID that the model deployment will belong to.

.. code-block:: python3

    from ads.model.deployment import ModelDeployer

    deployer = ModelDeployer()
    deployment = deployer.deploy(
        model_id="<MODEL_OCID>",
        display_name="Model Deployment Demo using ADS",
        instance_shape="VM.Standard.E4.Flex",
        instance_count=1,
        memory_in_gbs=6, # Applicable for the flexible shapes for example, “VM.Standard.E4.Flex”.
        ocpus=2, # Applicable for the flexible shapes for example, “VM.Standard.E4.Flex”.
        project_id="<PROJECT_OCID>",
        compartment_id="<COMPARTMENT_OCID>",
        # The following are optional
        access_log_group_id="<ACCESS_LOG_GROUP_OCID>",
        access_log_id="<ACCESS_LOG_OCID>",
        predict_log_group_id="<PREDICT_LOG_GROUP_OCID>",
        predict_log_id="<PREDICT_LOG_OCID>"
    )

