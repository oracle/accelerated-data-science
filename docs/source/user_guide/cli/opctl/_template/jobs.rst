++++++++++++++++++++++++++++++++++++++++++++
Working with OCI Data Science Jobs Using CLI
++++++++++++++++++++++++++++++++++++++++++++

Prerequisite
------------

.. include:: jobs_local_prerequisite.rst

Running a Pre Defined Job 
-------------------------

.. code-block:: shell

    aads opctl run -j <job ocid>

Delete Job or Job Run
---------------------

.. code-block:: shell

    ads opctl delete <job-id or run-id>

Cancel Job Run
--------------

.. code-block:: shell

    ads opctl cancel <run-id>

Cancel Distributed Training Job
-------------------------------

Stop a running cluster using ``cancel`` subcommand.

**Option 1: Using Job OCID and Work Dir**

.. code-block:: shell
  
  ads opctl cancel -j <job ocid> --work-dir <Object storage working directory specified when the cluster was created>

**Option 2: Using cluster info file**

Cluster info file is a yaml file with output generated from ``ads opctl run -f``

.. code-block:: shell
  
  ads opctl cancel -j <job ocid> --work-dir <Object storage working directory specified when the cluster was created>

This command requires an api key or resource principal setup. The logs are streamed from the logging service. If your job is not attached to logging service, this option will show only the lifecycle state.
