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

`with Proxy server:`

.. code-block:: bash

  docker build  --build-arg no_proxy=$(no_proxy) \
                --build-arg http_proxy=$(http_proxy) \
                --build-arg https_proxy=$(http_proxy) \
                -t $(IMAGE_NAME):$(TAG) \
                -f oci_dist_training_artifacts/horovod/v1/docker/<tensorflow.cpu.Dockerfile|tensorflow.gpu.Dockerfile> .

`without Proxy server:`

.. code-block:: bash

  docker build -t $(IMAGE_NAME):$(TAG) \
      -f oci_dist_training_artifacts/horovod/v1/docker/<tensorflow.cpu.Dockerfile|tensorflow.gpu.Dockerfile> .

Source code directory can be provided using the 'CODE_DIR' directory. In the following example, the code related
files are assumed to be in the 'code' directory (within the current directory).

.. code-block:: bash

  docker build --build-arg CODE_DIR=<code_folder> \
      -t $(IMAGE_NAME):$(TAG) \
      -f oci_dist_training_artifacts/horovod/v1/<tensorflow.cpu.Dockerfile|tensorflow.gpu.Dockerfile>

**Publish image to OCI Container Registry:**

You are now required to push the docker image in a OCI `container registry repo <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_ . First, tag the image using the following full name format
``<region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>``. Skip this, if you have already used this tag in the previous build command.

Tag

.. code-block:: bash

   docker tag <image_name>:<image_tag> <region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>

Push the image to OCI container registry.

.. code-block:: bash

   docker push <region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>

**Note:** You would need to login to OCI container registry. Refer `publishing images using the docker cli <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm>`_.

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

Submit the workload -

.. code-block:: bash

  ads opctl run -f train.yaml

Once running, you will see on the terminal an output similar to the below. Note that this yaml
can be used as input to ``ads opctl distributed-training show-config -f <info.yaml>`` - to both
save and see the run info use ``tee`` - for example:

.. code-block:: bash

  ads opctl run -f train.yaml | tee info.yaml

.. code-block:: yaml
  :caption: info.yaml

  jobId: oci.xxxx.<job_ocid>
  mainJobRunId: oci.xxxx.<job_run_ocid>
  workDir: oci://my-bucket@my-namespace/daskcluster-testing/005
  workerJobRunIds:
    - oci.xxxx.<job_run_ocid>
    - oci.xxxx.<job_run_ocid>
    - oci.xxxx.<job_run_ocid>

.. _hvd_saving_artifacts:

**Saving Artifacts to Object Storage Buckets**




In case you want to save the artifacts generated by the training process (model checkpoints, TensorBoard logs, etc.) to an object bucket
you can use the 'sync' feature. The environment variable ``OCI__SYNC_DIR`` exposes the directory location that will be automatically synchronized
to the configured object storage bucket location. Use this directory in your training script to save the artifacts.

To configure the destination object storage bucket location, use the following settings in the workload yaml file(train.yaml).

.. code-block:: bash

    - name: SYNC_ARTIFACTS
      value: 1
    - name: WORKSPACE
      value: "<bucket_name>"
    - name: WORKSPACE_PREFIX
      value: "<bucket_prefix>"

**Note**: Change ``SYNC_ARTIFACTS`` to ``0`` to disable this feature.
Use ``OCI__SYNC_DIR`` env variable in your code to save the artifacts. Example:

.. code-block:: python


  tf.keras.callbacks.ModelCheckpoint(os.path.join(os.environ.get("OCI__SYNC_DIR"),"ckpts",'checkpoint-{epoch}.h5'))


**Monitoring the workload logs**

To view the logs from a job run, you could run -

.. code-block:: bash

  ads jobs watch oci.xxxx.<job_run_ocid>

For more monitoring options, please refer to :doc:`Monitoring Horovod Training<monitoring>`
