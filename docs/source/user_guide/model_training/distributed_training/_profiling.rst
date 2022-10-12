
**Profiling using Nvidia Nsights**


`Nvidia Nsights <https://github.com/horovod/horovod/tree/master/examples/elastic/pytorch>`__. is a system wide profiling tool from Nvidia that can be used to profile Deep Learning workloads.

Nsights requires no change in your training code. This works on process level. You can enable this experimental feature(highlighted in bold) in your training setup via the following configuration in the runtime yaml file.


.. code-block:: bash

    - name: PROFILE
      value: 1
    - name: PROFILE_CMD
      value: ""nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o /opt/ml/nsight_report -x true""


Refer `this <https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options>`__ for nsys profile command options. You can modify the command within the ``PROFILE_CMD`` but remember this is all experimental. The profiling reports are generated per node. You need to download the reports to your computer manually or via the oci command.

.. code-block:: bash

    oci os object bulk-download \
      -ns <namespace> \
      -bn <bucket_name> \
      --download-dir /path/on/your/computer \
      --prefix path/on/bucket/<job_id>

To view the reports, you would need to install Nsight Systems app from `here <https://developer.nvidia.com/nsight-systems>`_. Thereafter, open the downloaded reports in the Nsight Systems app.