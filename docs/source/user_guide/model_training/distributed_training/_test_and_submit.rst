**Test Locally:**

Before submitting the workload to jobs, you can run it locally to test your code, dependencies, configurations etc.
With ``-b local`` flag, it uses a local backend. Further when you need to run this workload on odsc jobs, simply use ``-b job`` flag instead.

.. code-block:: bash

  ads opctl run -f train.yaml -b local

If your code requires to use any oci services (like object bucket), you need to mount oci keys from your local host machine onto the container. This is already done for you assuming the typical location of oci keys ``~/.oci``. You can modify it though, in-case you have keys at a different location. You need to do this in the config.ini file.

.. code-block:: bash

  oci_key_mnt = ~/.oci:/home/oci_dist_training/.oci

**Submit the workload:**



.. code-block:: bash

  ads opctl run -f train.yaml -b job

**Note:**: This will automatically push the docker image to the
OCI `container registry repo <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_ .

Once running, you will see on the terminal an output similar to the below. Note that this yaml
can be used as input to ``ads opctl distributed-training show-config -f <info.yaml>`` - to both
save and see the run info use ``tee`` - for example:

.. code-block:: bash

  ads opctl run -f train.yaml | tee info.yaml

.. code-block:: yaml
  :caption: info.yaml

  jobId: oci.xxxx.<job_ocid>
  mainJobRunId:
    mainJobRunIdName: oci.xxxx.<job_run_ocid>
  workDir: oci://my-bucket@my-namespace/daskcluster-testing/005
  otherJobRunIds:
    - workerJobRunIdName_1: oci.xxxx.<job_run_ocid>
    - workerJobRunIdName_2: oci.xxxx.<job_run_ocid>
    - workerJobRunIdName_3: oci.xxxx.<job_run_ocid>