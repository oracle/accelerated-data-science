============
IAM Policies
============

To unleash the full potential of quick actions, you might need to configure corresponding IAM policies.


Object Storage
~~~~~~~~~~~~~~

In order to store the results in an Oracle Cloud Infrastructure Object Storage bucket and retrieve source data from there, it may be necessary to set up specific policies for these actions. Find more details for writing policies to control access to Archive Storage, Object Storage, and Data Transfer on this `page <https://docs.oracle.com/en-us/iaas/Content/Identity/Reference/objectstoragepolicyreference.htm#Details_for_Object_Storage_Archive_Storage_and_Data_Transfer>`_. However every service like `Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_ and `Data Flow Applications <https://docs.oracle.com/en-us/iaas/data-flow/using/home.htm>`_ have their own policies to access Object Storage. It would be preferred to start from the  `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_ document, to understand the common conception of the Data Science policies.

Data Science Job
~~~~~~~~~~~~~~~~

If you're running quick actions within Oracle Cloud Infrastructure `Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_, ensure you have the appropriate :doc:`policies <../../jobs/policies>` in place to grant access and permissions. It is advisable to begin with the `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_ document to comprehend the fundamental concepts of Data Science policies.
