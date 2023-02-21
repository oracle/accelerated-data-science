Infrastructure
**************

The Data Science Job infrastructure is defined by a
:py:class:`~ads.jobs.builders.infrastructure.dsc_job.DataScienceJob` instance. For example:


.. code-block:: python3

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_compartment_id("<compartment_ocid>")
        .with_project_id("<project_ocid>")
        .with_subnet_id("<subnet_ocid>")
        .with_shape_name("VM.Standard.E3.Flex")
        # Shape config details are applicable only for the flexible shapes.
        .with_shape_config_details(memory_in_gbs=16, ocpus=1)
        # Minimum block storage size is 50
        .with_block_storage_size(50)
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
    )

When creating a :py:class:`~ads.jobs.builders.infrastructure.dsc_job.DataScienceJob` instance,
the following configurations are required:

* Compartment ID
* Project ID
* Compute Shape

The following configurations are optional:
* Block Storage Size, defaults to 50 (GB)
* Log Group ID
* Log ID

Using Configurations from Notebook
==================================

If you are creating a job from an OCI Data Science
`Notebook Session <https://docs.oracle.com/en-us/iaas/data-science/using/manage-notebook-sessions.htm>`_,
the same infrastructure configurations from the notebook session will be used as defaults.
You can initialize the :py:class:`~ads.jobs.builders.infrastructure.dsc_job.DataScienceJob`
with the logging configurations and override the other options as needed. For example:

.. code-block:: python3

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        # Use a GPU shape for the job,
        # regardless of the shape used by the notebook session
        .with_shape_name("VM.GPU2.1")
        # compartment ID, project ID, subnet ID and block storage will be
        # the same as the ones set in the notebook session
    )

Compute Shapes
==============

You can get a list of currently supported compute shapes by calling ``DataScienceJob.instance_shapes()``.
Additionally, you can get a list of shapes are available for fast launch by calling ``DataScienceJob.fast_launch_shapes()``
Specifying a fast launch shape will allow your job to start as fast as possible.

Networking
==========

Data Science Job offers two types of networking: default networking (managed egress) and custom networking.
Default networking allows job runs to access public internet through a NAT gateway and OCI service through
a service gateway, both are configured automatically. Custom networking requires you to specify a subnet ID.
You can control the network access through the subnet and security lists.

If you specified a subnet ID, your job will be configured to have custom networking.
Otherwise, default networking will be used. Note that when you are in a Data Science Notebook Session,
the same networking configuration is be used by default.
You can specify the networking manually by calling ``with_job_infrastructure_type()``.

Logging
=======

Logging is not required to create the job.
However, it is highly recommended to enable logging for debugging and monitoring purpose.

In the preceding example, both the log OCID and corresponding log group OCID are specified
with the ``DataScienceJob`` instance.
If your administrator configured the permission for you to search for logging resources,
you can skip specifying the log group OCID because ADS can automatically retrieve it.

If you specify only the log group OCID and no log OCID,
a new Log resource is automatically created within the log group to store the logs,
see also `ADS Logging <../logging/logging.html>`_.
