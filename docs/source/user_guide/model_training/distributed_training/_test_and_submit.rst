**Test Locally:**

Before submitting the workload to jobs, you can run it locally to test your code, dependencies, configurations etc.
With ``-b local`` flag, it uses a local backend. Further when you need to run this workload on OCI data science jobs, simply use ``-b job`` flag instead.

.. code-block:: bash

  ads opctl run -f train.yaml -b local

If your code requires to use any oci services (like object bucket), you need to mount oci keys from your local host machine onto the container. This is already done for you assuming the typical location of oci keys ``~/.oci``. You can modify it though, in-case you have keys at a different location. You need to do this in the config.ini file.

.. code-block:: bash

  oci_key_mnt = ~/.oci:/home/oci_dist_training/.oci

Note that the local backend requires the source code for your workload is available locally in the source folder specified in the ``config.ini`` file.
If you specified Git repository or OCI object storage location as source code location in your workflow YAML, please make sure you have a local copy available for local testing.

**Submit the workload:**

.. code-block:: bash

  ads opctl run -f train.yaml -b job

**Note:**: This will automatically push the docker image to the
OCI `container registry repo <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_ .

Once running, you will see on the terminal outputs similar to the below

.. code-block:: yaml
  :caption: info.yaml

  jobId: oci.xxxx.<job_ocid>
  mainJobRunId:
    mainJobRunIdName: oci.xxxx.<job_run_ocid>
  workDir: oci://my-bucket@my-namespace/cluster-testing/005
  otherJobRunIds:
    - workerJobRunIdName_1: oci.xxxx.<job_run_ocid>
    - workerJobRunIdName_2: oci.xxxx.<job_run_ocid>
    - workerJobRunIdName_3: oci.xxxx.<job_run_ocid>

This information can be saved as YAML file and used as input to ``ads opctl distributed-training show-config -f <info.yaml>``.
You can use ``--job-info`` to save the job run info into YAML, for example:

.. code-block:: bash

  ads opctl run -f train.yaml --job-info info.yaml
