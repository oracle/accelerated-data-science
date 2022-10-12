==============
Configurations
==============

--------
Networks
--------

You need to use a private subnet for distributed training and configure the security list to allow traffic through specific ports for communication between nodes. The following default ports are used by the corresponding frameworks:

* `Dask`:
  
  * ``Scheduler Port``: **8786**. More information `here <https://docs.dask.org/en/stable/deploying-cli.html#dask-scheduler>`_
  * ``Dashboard Port``: **8787**. More information `here <https://docs.dask.org/en/stable/deploying-cli.html#dask-scheduler>`_
  * ``Worker Ports``: **Default is Random**. It is good to open a specific range of port and then provide the value in the startup option. More information `here <https://docs.dask.org/en/stable/deploying-cli.html#dask-worker>`_
  * ``Nanny Process Ports``: **Default is Random**. It is good to open a specific range of port and then provide the value in the startup option. More information `here <https://docs.dask.org/en/stable/deploying-cli.html#dask-worker>`_

* `PyTorch`: By default, ``PyTorch`` uses **29400**.
* `Horovod`: allow TCP traffic on all ports within the subnet.
* `Tensorflow`: Worker Port: Allow traffic from all source ports to one worker port (default: 12345). If changed, provide this in train.yaml config.

See also: `Security Lists <https://docs.oracle.com/en-us/iaas/Content/Network/Concepts/securitylists.htm>`_


------------
OCI Policies
------------

Several OCI policies are needed for distributed training.

.. admonition:: Policy subject

  In the following example, ``group <your_data_science_users>`` is the subject of the policy. When starting the job from an OCI notebook session using resource principal, the subject should be ``dynamic-group``, for example, ``dynamic-group <your_notebook_sessions>``

Distributed training uses `OCI Container Registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_ to store the container image.

To push images to container registry, the ``manage repos`` policy is needed, for example:

.. code-block::

  Allow group <your_data_science_users> to manage repos in compartment <your_compartment_name>

To pull images from container registry for local testing, the ``use repos`` policy is needed, for example:

.. code-block::

  Allow group <your_data_science_users> to read repos in compartment <your_compartment_name>

You can also restrict the permission to specific repository, for example:

.. code-block::

  Allow group <your_data_science_users> to read repos in compartment <your_compartment_name> where all { target.repo.name=<your_repo_name> }

See also: `Policies to Control Repository Access <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registrypolicyrepoaccess.htm>`_

To start distributed training jobs, the user will need access to multiple resources, including:

* ``read repos``
* ``manage data-science-jobs``
* ``manage data-science-job-runs``
* ``use virtual-network-family``
* ``manage log-groups``
* ``use log-content``
* ``read metrics``

For example:

.. code-block::

  Allow group <your_data_science_users> to manage data-science-jobs in compartment <your_compartment_name>
  Allow group <your_data_science_users> to manage data-science-job-runs in compartment <your_compartment_name>
  Allow group <your_data_science_users> to use virtual-network-family in compartment <your_compartment_name>
  Allow group <your_data_science_users> to manage log-groups in compartment <your_compartment_name>
  Allow group <your_data_science_users> to use logging-family in compartment <your_compartment_name>
  Allow group <your_data_science_users> to use read metrics in compartment <your_compartment_name>

We also need policies for job runs, for example:

.. code-block::

  Allow dynamic-group <distributed_training_job_runs> to read repos in compartment <your_compartment_name>
  Allow dynamic-group <distributed_training_job_runs> to use data-science-family in compartment <your_compartment_name>
  Allow dynamic-group <distributed_training_job_runs> to use virtual-network-family in compartment <your_compartment_name>
  Allow dynamic-group <distributed_training_job_runs> to use log-groups in compartment <your_compartment_name>
  Allow dynamic-group <distributed_training_job_runs> to use logging-family in compartment <your_compartment_name>

See also `Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_.

Distributed training uses OCI Object Storage to store artifacts and outputs. The bucket should be created before starting any distributed training. The ``manage objects`` policy is needed for users and job runs to read/write files in the bucket. The ``manage buckets`` policy is required for job runs to synchronize generated artifacts. For example:

.. code-block::

  Allow group <your_data_science_users> to manage objects in compartment your_compartment_name where all {target.bucket.name=<your_bucket_name>}
  Allow dynamic-group <distributed_training_job_runs> to manage objects in compartment your_compartment_name where all {target.bucket.name=<your_bucket_name>}
  Allow dynamic-group <distributed_training_job_runs> to manage buckets in compartment your_compartment_name where all {target.bucket.name=<your_bucket_name>}

See also `Object Storage Policies <https://docs.oracle.com/en-us/iaas/Content/Identity/Reference/objectstoragepolicyreference.htm#Details_for_Object_Storage_Archive_Storage_and_Data_Transfer>`_

-------------
Policy Syntax
-------------

The overall syntax of a policy statement is as follows:

``Allow <subject> to <verb> <resource-type> in <location> where <conditions>``

See also: https://docs.oracle.com/en-us/iaas/Content/Identity/Concepts/policysyntax.htm

For ``<subject>``:

* If you are using API key authentication, ``<subject>`` should be the group your user belongs to. For example, ``group <your_data_science_users>``.
* If you are using resource principal or instance principal authentication, ``<subject>`` should be the dynamic group to which your OCI resource belongs. Here the resource is where you initialize the API requests, which is usually a job run, a notebook session or compute instance. For example, ``dynamic-group <distributed_training_job_runs>``

`Dynamic group <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/managingdynamicgroups.htm>`_ allows you to group OCI resources like job runs and notebook sessions. Distributed training is running on Data Science Jobs, for the training process to access resources, the job runs need to be defined as a dynamic group and use as the ``<subject>`` for policies.

In the following examples, we define ``distributed_training_job_runs`` dynamic group as:

``all { resource.type='datasciencejobrun', resource.compartment.id='<job_run_compartment_ocid>' }``

We also assume the user in ``group <your_data_science_users>`` is preparing the docker image and starting the training job.

The `<verb> <https://docs.oracle.com/en-us/iaas/Content/Identity/Reference/policyreference.htm#Verbs>`_ determines the ability of the <subject> to work on the ``<resource-type>``. Four options are available: inspect, read, user and manage.

The ``<resource-type>`` specifies the resources we would like to access. Distributed training uses the following OCI resources/services:

* `Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_. Resource Type: ``data-science-jobs`` and ``data-science-job-runs``
* `Object Storage <https://docs.oracle.com/en-us/iaas/Content/Identity/policyreference/objectstoragepolicyreference.htm#Details_for_Object_Storage_Archive_Storage_and_Data_Transfer>`_. Resource Type: ``buckets`` and ``objects``
* `Container Registry <https://docs.oracle.com/en-us/iaas/Content/Identity/policyreference/registrypolicyreference.htm#Details_for_Registry>`_. Resource Type: ``repos``

The ``<location>`` is usually the compartment or tenancy that your resources (specified by ``<resource-type>``) resides.
* If you would like the ``<subject>`` to have access to all resources (specified by ``<resource-type>``) in the tenancy, you can use ``tenancy`` as ``<location>``.
* If you would like the ``<subject>`` to have access to resources in specific compartment, you can use ``compartment your_compartment_name`` as ``<location>``.

The where ``<conditions>`` can be used to filter the resources specified in ``<resource-type>``.
