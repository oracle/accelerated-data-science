Creating Horovod Workloads
--------------------------

.. include:: ../_prerequisite.rst


.. _hvd_training_code:

**Write your training code:**

Your model training script (TensorFlow or PyTorch) needs to be adapted to use (Elastic) Horovod APIs for distributed training. Refer :doc:`Writing distributed code with horovod framework<coding>`

Also see : `Horovod Examples <https://github.com/horovod/horovod/tree/master/examples>`_

For this example, the code to run was inspired from an example
`found here <https://github.com/horovod/horovod/blob/master/examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py>`_ .
There are minimal changes to this script to save the training artifacts and TensorBoard logs to a folder referenced by
``OCI__SYNC_DIR`` environment variable. ``OCI__SYNC_DIR`` is a pre-provisioned folder which can be synchronized with an object bucket during the training process.


.. code-block:: python
  :caption: train.py
  :name: train.py

  # Script adapted from https://github.com/horovod/horovod/blob/master/examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py

  # ==============================================================================


  import argparse
  import tensorflow as tf
  import horovod.tensorflow.keras as hvd
  from distutils.version import LooseVersion

  import os

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

  parser = argparse.ArgumentParser(description="Tensorflow 2.0 Keras MNIST Example")

  parser.add_argument(
      "--use-mixed-precision",
      action="store_true",
      default=False,
      help="use mixed precision for training",
  )

  parser.add_argument(
      "--data-dir",
      help="location of the training dataset in the local filesystem (will be downloaded if needed)",
      default='/code/data/mnist.npz'
  )

  args = parser.parse_args()

  if args.use_mixed_precision:
      print(f"using mixed precision {args.use_mixed_precision}")
      if LooseVersion(tf.__version__) >= LooseVersion("2.4.0"):
          from tensorflow.keras import mixed_precision

          mixed_precision.set_global_policy("mixed_float16")
      else:
          policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
          tf.keras.mixed_precision.experimental.set_policy(policy)

  # Horovod: initialize Horovod.
  hvd.init()

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices("GPU")
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

  import numpy as np

  minist_local = args.data_dir


  def load_data():
      print("using pre-fetched dataset")
      with np.load(minist_local, allow_pickle=True) as f:
          x_train, y_train = f["x_train"], f["y_train"]
          x_test, y_test = f["x_test"], f["y_test"]
          return (x_train, y_train), (x_test, y_test)


  (mnist_images, mnist_labels), _ = (
      load_data()
      if os.path.exists(minist_local)
      else tf.keras.datasets.mnist.load_data(path="mnist-%d.npz" % hvd.rank())
  )


  dataset = tf.data.Dataset.from_tensor_slices(
      (
          tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
          tf.cast(mnist_labels, tf.int64),
      )
  )
  dataset = dataset.repeat().shuffle(10000).batch(128)

  model = tf.keras.Sequential(
      [
          tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
          tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Dropout(0.25),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(10, activation="softmax"),
      ]
  )

  # Horovod: adjust learning rate based on number of GPUs.
  scaled_lr = 0.001 * hvd.size()
  opt = tf.optimizers.Adam(scaled_lr)

  # Horovod: add Horovod DistributedOptimizer.
  opt = hvd.DistributedOptimizer(
      opt, backward_passes_per_step=1, average_aggregated_gradients=True
  )

  # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
  # uses hvd.DistributedOptimizer() to compute gradients.
  model.compile(
      loss=tf.losses.SparseCategoricalCrossentropy(),
      optimizer=opt,
      metrics=["accuracy"],
      experimental_run_tf_function=False,
  )

  # Horovod: initialize optimizer state so we can synchronize across workers
  # Keras has empty optimizer variables() for TF2:
  # https://sourcegraph.com/github.com/tensorflow/tensorflow@v2.4.1/-/blob/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L351:10
  model.fit(dataset, steps_per_epoch=1, epochs=1, callbacks=None)

  state = hvd.elastic.KerasState(model, batch=0, epoch=0)


  def on_state_reset():
      tf.keras.backend.set_value(state.model.optimizer.lr, 0.001 * hvd.size())
      # Re-initialize, to join with possible new ranks
      state.model.fit(dataset, steps_per_epoch=1, epochs=1, callbacks=None)


  state.register_reset_callbacks([on_state_reset])

  callbacks = [
      hvd.callbacks.MetricAverageCallback(),
      hvd.elastic.UpdateEpochStateCallback(state),
      hvd.elastic.UpdateBatchStateCallback(state),
      hvd.elastic.CommitStateCallback(state),
  ]

  # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
  # save the artifacts in the OCI__SYNC_DIR dir.
  artifacts_dir = os.environ.get("OCI__SYNC_DIR") + "/artifacts"
  tb_logs_path = os.path.join(artifacts_dir, "logs")
  check_point_path = os.path.join(artifacts_dir, "ckpts", "checkpoint-{epoch}.h5")
  if hvd.rank() == 0:
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path))
      callbacks.append(tf.keras.callbacks.TensorBoard(tb_logs_path))

  # Train the model.
  # Horovod: adjust number of steps based on number of GPUs.
  @hvd.elastic.run
  def train(state):
      state.model.fit(
          dataset,
          steps_per_epoch=500 // hvd.size(),
          epochs=2 - state.epoch,
          callbacks=callbacks,
          verbose=1,
      )


  train(state)



**Initialize a distributed-training folder:**

At this point you have created a training file (or files) - ``train.py`` from the above
example. Now, run the command below.

.. code-block:: bash

  ads opctl distributed-training init --framework horovod-tensorflow --version v1

**Note**: If you choose to run a PyTorch example instead, use ``horovod-pytorch`` as the framework.

.. code-block:: bash

  ads opctl distributed-training init --framework horovod-pytorch --version v1

This will download the ``horovod-tensorflow|horovod-pytorch`` framework and place it inside ``'oci_dist_training_artifacts'`` folder.

**Containerize your code and build container:**

To build the image:

Horovod frameworks for TensorFlow and PyTorch contains two separate docker files, for cpu and gpu. Choose the docker file
based on whether you are going to use cpu or gpu based shapes.

Before you can build the image, you must set the following environment variables:

Specify image name and tag

.. code-block:: bash

  export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
  export TAG=latest

Build the container image.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/horovod/v1/<pytorch|tensorflow>.<cpu|gpu>.Dockerfile

The code is assumed to be in the current working directory. To override the source code directory, use the ``-s`` flag and specify the code dir. This folder should be within the current working directory.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
       -df oci_dist_training_artifacts/horovod/v1/<pytorch|tensorflow>.<cpu|gpu>.Dockerfile
      -s <code_dir>

If you are behind proxy, ads opctl will automatically use your proxy settings (defined via ``no_proxy``, ``http_proxy`` and ``https_proxy``).


**SSH Setup:**

In Horovod distributed training, communication between scheduler and worker(s) uses a secure connection. For this purpose, SSH keys need to be provisioned in the scheduler and worker nodes.
This is already taken care in the docker images. When the docker image is built, SSH key pair is placed inside the image with required configuration changes (adding public key to authorized_keys file).
This enables a secure connection between scheduler and the workers.

**Define your workload yaml:**

The ``yaml`` file is a declarative way to express the workload.

.. code-block:: yaml
  :caption: train.yaml
  :name: train.yaml

  kind: distributed
  apiVersion: v1.0
  spec:
    infrastructure: # This section maps to Job definition. Does not include environment variables
      kind: infrastructure
      type: dataScienceJob
      apiVersion: v1.0
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: HVD-Distributed-TF
        logGroupId: oci.xxxx.<log_group_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.GPU2.1
        blockStorageSize: 50
    cluster:
      kind: HOROVOD
      apiVersion: v1.0
      spec:
        image: "<region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>"
        workDir:  "oci://<bucket_name>@<bucket_namespace>/<bucket_prefix>"
        name: "horovod_tf"
        config:
          env:
            # MIN_NP, MAX_NP and SLOTS are inferred from the shape. Modify only when needed.
            # - name: MIN_NP
            #   value: 2
            # - name: MAX_NP
            #   value: 4
            # - name: SLOTS
            #   value: 2
            - name: WORKER_PORT
              value: 12345
            - name: START_TIMEOUT #Optional: Defaults to 600.
              value: 600
            - name: ENABLE_TIMELINE # Optional: Disabled by Default.Significantly increases training duration if switched on (1).
              value: 0
            - name: SYNC_ARTIFACTS #Mandatory: Switched on by Default.
              value: 1
            - name: WORKSPACE #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket to sync generated artifacts to.
              value: "<bucket_name>"
            - name: WORKSPACE_PREFIX #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket folder to sync generated artifacts to.
              value: "<bucket_prefix>"
            - name: HOROVOD_ARGS # Parameters for cluster tuning.
              value: "--verbose"
        main:
          name: "scheduler"
          replicas: 1 #this will be always 1
        worker:
          name: "worker"
          replicas: 2 #number of workers
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: "/code/train.py" #location of user's training script in docker image.
        args:  #any arguments that the training script requires.
        env:

**Use ads opctl to create the cluster infrastructure and run the workload:**

Do a dry run to inspect how the yaml translates to Job and Job Runs

.. code-block:: bash

  ads opctl run -f train.yaml --dry-run

This will give output similar to this.

.. code-block:: bash

    -----------------------------Entering dryrun mode----------------------------------
  Creating Job with payload:
  kind: job
  spec:
    infrastructure:
      kind: infrastructure
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: HVD-Distributed-TF
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.GPU2.1
        blockStorageSize: 50
      type: dataScienceJob
    name: horovod_tf
    runtime:
      kind: runtime
      spec:
        entrypoint: null
        env:
        - name: WORKER_PORT
          value: 12345
        - name: START_TIMEOUT
          value: 600
        - name: ENABLE_TIMELINE
          value: 0
        - name: SYNC_ARTIFACTS
          value: 1
        - name: WORKSPACE
          value: "<bucket_name>"
        - name: WORKSPACE_PREFIX
          value: "<bucket_prefix>"
        - name: HOROVOD_ARGS
          value: --verbose
        - name: OCI__WORK_DIR
          value: oci://<bucket_name>@<bucket_namespace>/<bucket_prefix>
        - name: OCI__EPHEMERAL
          value: None
        - name: OCI__CLUSTER_TYPE
          value: HOROVOD
        - name: OCI__WORKER_COUNT
          value: '2'
        - name: OCI__START_ARGS
          value: ''
        - name: OCI__ENTRY_SCRIPT
          value: /code/train.py
        image: "<region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>"
      type: container

  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Creating Main Job with following details:
  Name: scheduler
  Environment Variables:
      OCI__MODE:MAIN
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Creating 2 worker jobs with following details:
  Name: worker
  Environment Variables:
      OCI__MODE:WORKER
  -----------------------------Ending dryrun mode----------------------------------

.. include:: ../_test_and_submit.rst

.. _hvd_saving_artifacts:

.. include:: ../_save_artifacts.rst

.. code-block:: python

  tf.keras.callbacks.ModelCheckpoint(os.path.join(os.environ.get("OCI__SYNC_DIR"),"ckpts",'checkpoint-{epoch}.h5'))

**Monitoring the workload logs**

To view the logs from a job run, you could run -

.. code-block:: bash

  ads jobs watch oci.xxxx.<job_run_ocid>

For more monitoring options, please refer to :doc:`Monitoring Horovod Training<monitoring>`

