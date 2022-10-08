Properties
**********

``ModelDeploymentProperties``
=============================

The ``ModelDeploymentProperties`` class is a container to store model deployment properties. String properties are set using the ``.with_prop()`` method. You use it to assemble properties such as the display name, project OCID, and compartment OCID. The ``.with_access_log()`` and ``.with_predict_log()`` methods define the logging properties. Alternatively, you could use the ``.with_logging_configuration()`` helper method to define the predict and access log properties using a single method. The ``.with_instance_configuration()`` method defines the instance shape, count, and bandwidth.  Initializing ``ModelDeploymentProperties`` requires a ``model_id`` or ``model_uri``.  The ``model_id`` is the model OCID from the model catalog.

.. code-block:: python3

    from ads.model.deployment import ModelDeploymentProperties

    model_deployment_properties = ModelDeploymentProperties(
        "<MODEL_OCID>"
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

Alternatively, you can specify a ``model_uri`` instead of a ``model_id``. The
``model_uri`` is the path to the directory containing the model artifact. This can be a local path or
the URI of Object Storage. For example, ``oci://your_bucket@your_namespace/path/to/dir``.

.. code-block:: python3

    model_deployment_properties = ModelDeploymentProperties(
       "<oci://your_bucket@your_namespace/path/to/dir>"
    )


``properties``
==============

The ``ModelDeployment`` class has a number of attributes that provide information about the deployment. The ``properties`` attribute contains information about the model deployment’s properties that are related to the information that is stored in the model's ``ModelDeploymentProperties`` object. This object has all of the attributes of the `Data Science model deployment model <https://oracle-cloud-infrastructure-python-sdk.readthedocs.io/en/latest/api/data_science/models/oci.data_science.models.ModelDeployment.html#oci.data_science.models.ModelDeployment>`__.  The most commonly used properties are:

*  ``category_log_details``: A model object that contains the OCIDs for the access and predict logs.
*  ``compartment_id``: Compartment ID of the model deployment.
*  ``created_by``: OCID of the user that created the model deployment.
*  ``defined_tags``: System defined tags.
*  ``description``: Description of the model deployment.
*  ``display_name``: Name of the model that is displayed in the Console.
*  ``freeform_tags``: User-defined tags.
*  ``instance_count``: The number of instances to deploy.
*  ``instance_shape``: The instance compute shape to use. For example, “VM.Standard2.1”.
*  ``memory_in_gbs``:  The size of the memory of the model deployment instance in GBs. Applicable for the flexible shapes, for example, “VM.Standard.E4.Flex”.
*  ``model_id``: OCID of the deployed model.
*  ``ocpus``: The ocpus count of the model deployment instance. Applicable for the flexible shapes, for example, “VM.Standard.E4.Flex”.
*  ``project_id``: OCID of the project the model deployment belongs to.

To access these properties use the ``.properties`` accessor on a ``ModelDeployment`` object.  For example, to determine the OCID of the project that a model deployment is associated with, use the command:

.. code-block:: python3

    deployment.properties.project_id


