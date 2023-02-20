I AM Policies
*************

Oracle Cloud Infrastructure Identity and Access Management (IAM)
lets you specify policies to control the access to your cloud resources.
This section contain the policies recommended for Data Science Jobs.

.. admonition:: Policy subject

    In the following example, ``group <your_data_science_users>`` is the subject of the policy
    when using OCI API keys for authentication. For resource principal authentication,
    the subject should be a ``dynamic-group``, for example, ``dynamic-group <your_resources>``

Here is an example defining a dynamic group for all job runs in a compartment:

.. code-block::

    all { resource.type='datasciencejobrun', resource.compartment.id='<job_run_compartment_ocid>' }

The following policies are for creating and managing Data Science Jobs and Job Runs:

.. code-block::

    Allow group <your_data_science_users> to manage data-science-jobs in compartment <your_compartment_name>
    Allow group <your_data_science_users> to manage data-science-job-runs in compartment <your_compartment_name>
    Allow group <your_data_science_users> to use virtual-network-family in compartment <your_compartment_name>
    Allow group <your_data_science_users> to manage log-groups in compartment <your_compartment_name>
    Allow group <your_data_science_users> to use logging-family in compartment <your_compartment_name>
    Allow group <your_data_science_users> to use read metrics in compartment <your_compartment_name>

The following policies are for job runs to access other OCI resources:

.. code-block::

    Allow dynamic-group <your_resources> to read repos in compartment <your_compartment_name>
    Allow dynamic-group <your_resources> to use data-science-family in compartment <your_compartment_name>
    Allow dynamic-group <your_resources> to use virtual-network-family in compartment <your_compartment_name>
    Allow dynamic-group <your_resources> to use log-groups in compartment <your_compartment_name>
    Allow dynamic-group <your_resources> to use logging-family in compartment <your_compartment_name>
    Allow dynamic-group <your_resources> to manage objects in compartment <your_compartment_name> where all {target.bucket.name=<your_bucket_name>}
    Allow dynamic-group <your_resources> to use buckets in compartment <your_compartment_name> where all {target.bucket.name=<your_bucket_name>}

The following policy is needed for running a container job:

.. code-block::

    Allow dynamic-group <your_resources> to read repos in compartment <your_compartment_name>

See also:

* `Dynamic Group <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/managingdynamicgroups.htm>`_
* `Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_
* `Object Storage <https://docs.oracle.com/en-us/iaas/Content/Identity/Reference/objectstoragepolicyreference.htm#Details_for_Object_Storage_Archive_Storage_and_Data_Transfer>`_
* `Container Registry <https://docs.oracle.com/en-us/iaas/Content/Identity/policyreference/registrypolicyreference.htm#Details_for_Registry>`_
