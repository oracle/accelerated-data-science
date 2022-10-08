Monitoring Training
-------------------

Monitoring Horovod training using TensorBoard is similar to how it is usually done for TensorFlow
or PyTorch workloads. Your training script generates the TensorBoard logs and saves the logs to
the directory reference by ``OCI__SYNC_DIR`` env variable. With ``SYNC_ARTIFACTS=1``, these TensorBoard logs will
be periodically synchronized with the configured object storage bucket.

Please refer :ref:`Saving Artifacts to Object Storage Buckets <hvd_saving_artifacts>`.


**Aggregating metrics:**

In a distributed setup, the metrics(loss, accuracy etc.) need to be aggregated from all the workers. Horovod provides
`MetricAverageCallback <https://horovod.readthedocs.io/en/stable/_modules/horovod/tensorflow/keras/callbacks.html>`_ callback(for TensorFlow) which should be added to the model training step.
For PyTorch, refer this `Pytorch Example <https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py>`_.

**Using TensorBoard Logs:**

TensorBoard can be setup on a local machine and pointed to object storage. This will enable a live monitoring setup
of TensorBoard logs.

.. code-block:: bash

   OCIFS_IAM_TYPE=api_key tensorboard --logdir oci://<bucket_name>/path/to/logs


**Note**: The logs take some initial time (few minutes) to reflect on the tensorboard dashboard.

**Horovod Timelines:**

Horovod also provides `Timelines <https://horovod.readthedocs.io/en/stable/timeline_include.html>`_, which
provides a snapshot of the training activities. Timeline files can be optionally generated with the
following environment variable(part of workload yaml).

.. code-block:: yaml

      config:
          env:
            - name: ENABLE_TIMELINE #Disabled by Default(0).
              value: 1

**Note**: Creating Timelines degrades the training execution time.
