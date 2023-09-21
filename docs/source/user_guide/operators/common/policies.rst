============
IAM Policies
============

To unleash the full potential of operators, you might need to configure corresponding IAM policies.


Object Storage
~~~~~~~~~~~~~~

In order to store the results in an Oracle Cloud Infrastructure Object Storage bucket and retrieve source data from there, it may be necessary to set up specific policies for these actions. Find more details for writing policies to control access to Archive Storage, Object Storage, and Data Transfer on this `page <https://docs.oracle.com/en-us/iaas/Content/Identity/Reference/objectstoragepolicyreference.htm#Details_for_Object_Storage_Archive_Storage_and_Data_Transfer>`_. However every service like `Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_ and `Data Flow Applications <https://docs.oracle.com/en-us/iaas/data-flow/using/home.htm>`_ have their own policies to access Object Storage. It would be preferred to start from the  `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_ document, to understand the common conception of the Data Science policies.


Oracle Container Registry
~~~~~~~~~~~~~~~~~~~~~~~~~

`Oracle Cloud Infrastructure Registry <https://docs.oracle.com/en-us/iaas/Content/Registry/home.htm>`_ (also known as Container Registry) is an Oracle-managed registry that enables you to simplify your development to production workflow. To facilitate the publication of an operator's containers to the Oracle Container Registry, you may be required to configure the authentication `token <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm#Pushing_Images_Using_the_Docker_CLI>`_ for this purpose.


Data Science Job
~~~~~~~~~~~~~~~~

If you're running operators within Oracle Cloud Infrastructure `Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_, ensure you have the appropriate :doc:`policies <../../jobs/policies>` in place to grant access and permissions. It is advisable to begin with the `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_ document to comprehend the fundamental concepts of Data Science policies.


Data Flow Application
~~~~~~~~~~~~~~~~~~~~~

Oracle Cloud Infrastructure `Data Flow <https://docs.oracle.com/en-us/iaas/data-flow/using/home.htm>`_ is a fully managed service for running Apache Spark ™ applications, offering a simplified runtime environment for execution. Data Flow can serve as one of the backends for operators. However, `Data Flow <https://docs.oracle.com/en-us/iaas/data-flow/using/home.htm>`_ requires IAM policies to access resources for managing and running sessions. Refer to the `Data Flow Studio Policies <https://docs.oracle.com/en-us/iaas/data-flow/using/set-up-iam-policies.htm>`_ documentation for guidance on policy setup.

After configuring the core Data Flow policies, consult the `Policies Required to Integrate Data Flow and Data Science <https://docs.oracle.com/en-us/iaas/data-flow/using/policies-data-flow-studio.htm#policies-data-flow-studio>`_ documentation to enable Data Flow to write data to the Object Storage bucket and manage logs effectively.

To set up Object Storage for Data Flow, follow the `Set Up Object Store <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_object_store_setting_up_storage.htm>`_ documentation.
