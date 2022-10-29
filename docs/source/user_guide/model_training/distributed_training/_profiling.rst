
**Profiling using Nvidia Nsights**


`Nvidia Nsights <https://github.com/horovod/horovod/tree/master/examples/elastic/pytorch>`__. is a system wide profiling tool from Nvidia that can be used to profile Deep Learning workloads.

Nsights requires no change in your training code. This works on process level. You can enable this experimental feature in your training setup via the following configuration in the runtime yaml file(highlighted).


.. code-block:: bash
   :emphasize-lines: 15,16,17,18

    spec:
          image: "@image"
          workDir:  "oci://@/"
          name: "tf_multiworker"
          config:
            env:
              - name: WORKER_PORT
                value: 12345
              - name: SYNC_ARTIFACTS
                value: 1
              - name: WORKSPACE
                value: "<bucket_name>"
              - name: WORKSPACE_PREFIX
                value: "<bucket_prefix>"
              - name: PROFILE
                value: 1
              - name: PROFILE_CMD
                value: "nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o /opt/ml/nsight_report -x true"
          main:
            name: "main"
            replicas: 1
          worker:
            name: "worker"
            replicas: 1


Refer `this <https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options>`__ for nsys profile command options. You can modify the command within the ``PROFILE_CMD`` but remember this is all experimental. The profiling reports are generated per node. You need to download the reports to your computer manually or via the oci command.

.. code-block:: bash

    oci os object bulk-download \
      -ns <namespace> \
      -bn <bucket_name> \
      --download-dir /path/on/your/computer \
      --prefix path/on/bucket/<job_id>

**Note:** ``-bn``  == ``WORKSPACE`` and ``--prefix path`` == ``WORKSPACE_PREFIX/<job_id>`` , as configured in the runtime yaml file.
To view the reports, you would need to install Nsight Systems app from `here <https://developer.nvidia.com/nsight-systems>`_. Thereafter, open the downloaded reports in the Nsight Systems app.