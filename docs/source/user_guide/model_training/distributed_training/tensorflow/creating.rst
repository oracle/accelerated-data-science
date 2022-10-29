Creating Tensorflow Workloads
-----------------------------

.. include:: ../_prerequisite.rst


**Write your training code:**

Your model training script needs to use one of Distributed Strategies in tensorflow.

For example, you can have the following training Tensorflow script for MultiWorkerMirroredStrategy saved as `mnist.py`:

.. code-block:: python

  # Script adapted from tensorflow tutorial: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
  import tensorflow as tf
  import tensorflow_datasets as tfds
  import os
  import sys
  import time
  import ads
  from ocifs import OCIFileSystem
  from tensorflow.data.experimental import AutoShardPolicy

  BUFFER_SIZE = 10000
  BATCH_SIZE_PER_REPLICA = 64

  if '.' not in sys.path:
      sys.path.insert(0, '.')


  def create_dir(dir):
      if not os.path.exists(dir):
          os.makedirs(dir)


  def create_dirs(task_type="worker", task_id=0):
      artifacts_dir = os.environ.get("OCI__SYNC_DIR", "/opt/ml")
      model_dir = artifacts_dir + "/model"
      print("creating dirs for Model: ", model_dir)
      create_dir(model_dir)
      checkpoint_dir = write_filepath(artifacts_dir, task_type, task_id)
      return artifacts_dir, checkpoint_dir, model_dir

  def write_filepath(artifacts_dir, task_type, task_id):
      if task_type == None:
          task_type = "worker"
      checkpoint_dir = artifacts_dir + "/checkpoints/" + task_type + "/" + str(task_id)
      print("creating dirs for Checkpoints: ", checkpoint_dir)
      create_dir(checkpoint_dir)
      return checkpoint_dir


  def scale(image, label):
      image = tf.cast(image, tf.float32)
      image /= 255
      return image, label


  def get_data(data_bckt=None, data_dir="/code/data", num_replicas=1, num_workers=1):
      if data_bckt is not None and not os.path.exists(data_dir + '/mnist'):
          print(f"downloading data from {data_bckt}")
          ads.set_auth(os.environ.get("OCI_IAM_TYPE", "resource_principal"))
          authinfo = ads.common.auth.default_signer()
          oci_filesystem = OCIFileSystem(**authinfo)
          lck_file = os.path.join(data_dir, '.lck')
          if not os.path.exists(lck_file):
              os.makedirs(os.path.dirname(lck_file), exist_ok=True)
              open(lck_file, 'w').close()
              oci_filesystem.download(data_bckt, data_dir, recursive=True)
          else:
              print(f"data downloaded by a different process. waiting")
              time.sleep(30)

      BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas * num_workers
      print("Now printing data_dir:", data_dir)
      datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True, data_dir=data_dir)
      mnist_train, mnist_test = datasets['train'], datasets['test']
      print("num_train_examples :", info.splits['train'].num_examples, " num_test_examples: ",
            info.splits['test'].num_examples)

      train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
      test_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
      train = shard(train_dataset)
      test = shard(test_dataset)
      return train, test, info


  def shard(dataset):
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
      return dataset.with_options(options)


  def decay(epoch):
      if epoch < 3:
          return 1e-3
      elif epoch >= 3 and epoch < 7:
          return 1e-4
      else:
          return 1e-5


  def get_callbacks(model, checkpoint_dir="/opt/ml/checkpoints"):
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

      class PrintLR(tf.keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs=None):
              print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()), flush=True)

      callbacks = [
          tf.keras.callbacks.TensorBoard(log_dir='./logs'),
          tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                             # save_weights_only=True
                                             ),
          tf.keras.callbacks.LearningRateScheduler(decay),
          PrintLR()
      ]
      return callbacks


  def build_and_compile_cnn_model():
      print("TF_CONFIG in model:", os.environ.get("TF_CONFIG"))
      model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(10)
      ])

      model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])
      return model

And, save the following script as `train.py`


.. code-block:: python

  import tensorflow as tf
  import argparse
  import mnist
  
  print(tf.__version__)

  parser = argparse.ArgumentParser(description='Tensorflow Native MNIST Example')
  parser.add_argument('--data-dir',
                      help='location of the training dataset in the local filesystem (will be downloaded if needed)',
                      default='/code/data')
  parser.add_argument('--data-bckt',
                      help='location of the training dataset in an object storage bucket',
                      default=None)

  args = parser.parse_args()

  artifacts_dir, checkpoint_dir, model_dir = mnist.create_dirs()

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  train_dataset, test_dataset, info = mnist.get_data(data_bckt=args.data_bckt, data_dir=args.data_dir,
                                                     num_replicas=strategy.num_replicas_in_sync)
  with strategy.scope():
      model = mnist.build_and_compile_cnn_model()

  model.fit(train_dataset, epochs=2, callbacks=mnist.get_callbacks(model, checkpoint_dir))

  model.save(model_dir, save_format='tf')


**Initialize a distributed-training folder:**

At this point you have created a training file (or files) - ``train.py`` from the above
example. Now, run the command below.

.. code-block:: bash

  ads opctl distributed-training init --framework tensorflow --version v1


This will download the ``tensorflow`` framework and place it inside ``'oci_dist_training_artifacts'`` folder.

**Note**: Whenever you change the code, you have to build, tag and push the image to repo. This is automatically done in ```ads opctl run``` cli command.

**Containerize your code and build container:**

The required python dependencies are provided inside the conda environment file `oci_dist_training_artifacts/tensorflow/v1/environments.yaml`.  If your code requires additional dependency, update this file.

Also, while updating `environments.yaml` do not remove the existing libraries. You can append to the list.

Update the TAG and the IMAGE_NAME as per your needs -

.. code-block:: bash

  export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
  export TAG=latest
  export MOUNT_FOLDER_PATH=.

Build the container image.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/tensorflow/v1/Dockerfile \

The code is assumed to be in the current working directory. To override the source code directory, use the ``-s`` flag and specify the code dir. This folder should be within the current working directory.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/tensorflow/v1/Dockerfile \
      -s $MOUNT_FOLDER_PATH

If you are behind proxy, ads opctl will automatically use your proxy settings (defined via ``no_proxy``, ``http_proxy`` and ``https_proxy``).


**Define your workload yaml:**

The ``yaml`` file is a declarative way to express the workload.
In this example, we bring up 1 worker node and 1 chief-worker node.
The training code to run is ``train.py``.
All your training code is assumed to be present inside ``/code`` directory within the container.
Additionally, you can also put any data files inside the same directory
(and pass on the location ex ``/code/data/**`` as an argument to your training script using runtime->spec->args).


.. code-block:: yaml

  kind: distributed
  apiVersion: v1.0
  spec:
    infrastructure:
      kind: infrastructure
      type: dataScienceJob
      apiVersion: v1.0
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: Tensorflow
        logGroupId: oci.xxxx.<log_group_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.GPU2.1
        blockStorageSize: 50
    cluster:
      kind: TENSORFLOW
      apiVersion: v1.0
      spec:
        image: "@image"
        workDir:  "oci://<bucket_name>@<bucket_namespace>/<bucket_prefix>"
        name: "tf_multiworker"
        config:
          env:
            - name: WORKER_PORT #Optional. Defaults to 12345
              value: 12345
            - name: SYNC_ARTIFACTS #Mandatory: Switched on by Default.
              value: 1
            - name: WORKSPACE #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket to sync generated artifacts to.
              value: "<bucket_name>"
            - name: WORKSPACE_PREFIX #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket folder to sync generated artifacts to.
              value: "<bucket_prefix>"
        main:
          name: "chief"
          replicas: 1 #this will be always 1.
        worker:
          name: "worker"
          replicas: 1 #number of workers. This is in addition to the 'chief' worker. Could be more than 1
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: "/code/train.py" #location of user's training script in the container image.
        args:  #any arguments that the training script requires.
            - --data-dir    # assuming data folder has been bundled in the container image.
            - /code/data/
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
        displayName: Tensorflow
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.GPU2.1
        blockStorageSize: 50
      type: dataScienceJob
    name: tf_multiworker
    runtime:
      kind: runtime
      spec:
        entrypoint: null
        env:
        - name: WORKER_PORT
          value: 12345
        - name: SYNC_ARTIFACTS
          value: 1
        - name: WORKSPACE
          value: "<bucket_name>"
        - name: WORKSPACE_PREFIX
          value: "<bucket_prefix>"
        - name: OCI__WORK_DIR
          value: oci://<bucket_name>@<bucket_namespace>/<bucket_prefix>
        - name: OCI__EPHEMERAL
          value: None
        - name: OCI__CLUSTER_TYPE
          value: TENSORFLOW
        - name: OCI__WORKER_COUNT
          value: '1'
        - name: OCI__START_ARGS
          value: ''
        - name: OCI__ENTRY_SCRIPT
          value: /code/train.py
        image: "<region>.ocir.io/<tenancy_id>/<repo_name>/<image_name>:<image_tag>"
      type: container

  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Creating Main Job Run with following details:
  Name: chief
  Environment Variables:
      OCI__MODE:MAIN
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Creating Job Runs with following details:
  Name: worker_0
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

**Profiling**

You may want to profile your training setup for optimization/performance tuning. Profiling typically provides a detailed analysis of cpu utilization, gpu utilization,
top cuda kernels, top operators etc. You can choose to profile your training setup using the native Pytorch profiler or using a third party profiler such as `Nvidia Nsights <https://developer.nvidia.com/nsight-systems>`__.

**Profiling using Tensorflow Profiler**

`Tensorflow Profiler <https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras>`_ is a native offering from Tensforflow for Tensorflow performance profiling.

Profiling is invoked using code instrumentation using one of the following apis.

    `tf.keras.callbacks.TensorBoard <https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras>`_

    `tf.profiler.experimental.Profile <https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/Profile>`_

Refer above links for changes that you need to do in your training script for instrumentation.

You should choose the ``OCI__SYNC_DIR`` directory to save the profiling logs. For example:

.. code-block:: python

   options = tf.profiler.experimental.ProfilerOptions(
     host_tracer_level=2,
     python_tracer_level=1,
     device_tracer_level=1,
     delay_ms=None)
   with tf.profiler.experimental.Profile(os.environ.get("OCI__SYNC_DIR") + "/logs",options=options):
      # training code

In case of keras callback:

.. code-block:: python

   tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = os.environ.get("OCI__SYNC_DIR") + "/logs",
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')
   model.fit(...,callbacks = [tboard_callback])

Also, the sync feature ``SYNC_ARTIFACTS`` should be enabled ``'1'`` to sync the profiling logs to the configured object storage.

Thereafter, use Tensorboard to view logs. Refer the :doc:`Tensorboard setup <../../tensorboard/tensorboard>` for set-up on your computer.

.. include:: ../_profiling.rst

**Other Tensorflow Strategies supported**

Tensorflow has two multi-worker strategies: ``MultiWorkerMirroredStrategy`` and ``ParameterServerStrategy``.
Let's see changes that you would need to do to run ``ParameterServerStrategy`` workload.

You can have the following training Tensorflow script for ``ParameterServerStrategy`` saved as ``train.py``
(just like ``mnist.py`` and ``train.py`` in case of ``MultiWorkerMirroredStrategy``):


.. code-block:: python

  # Script adapted from tensorflow tutorial: https://www.tensorflow.org/tutorials/distribute/parameter_server_training

  import os
  import tensorflow as tf
  import json
  import multiprocessing

  NUM_PS = len(json.loads(os.environ['TF_CONFIG'])['cluster']['ps'])
  global_batch_size = 64


  def worker(num_workers, cluster_resolver):
      # Workers need some inter_ops threads to work properly.
      worker_config = tf.compat.v1.ConfigProto()
      if multiprocessing.cpu_count() < num_workers + 1:
          worker_config.inter_op_parallelism_threads = num_workers + 1

      for i in range(num_workers):
          print("cluster_resolver.task_id: ", cluster_resolver.task_id, flush=True)

          s = tf.distribute.Server(
              cluster_resolver.cluster_spec(),
              job_name=cluster_resolver.task_type,
              task_index=cluster_resolver.task_id,
              config=worker_config,
              protocol="grpc")
          s.join()


  def ps(num_ps, cluster_resolver):
      print("cluster_resolver.task_id: ", cluster_resolver.task_id, flush=True)
      for i in range(num_ps):
          s = tf.distribute.Server(
              cluster_resolver.cluster_spec(),
              job_name=cluster_resolver.task_type,
              task_index=cluster_resolver.task_id,
              protocol="grpc")
          s.join()


  def create_cluster(cluster_resolver, num_workers=1, num_ps=1, mode="worker"):
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      if mode.lower() == 'worker':
          print("Starting worker server...", flush=True)
          worker(num_workers, cluster_resolver)
      else:
          print("Starting ps server...", flush=True)
          ps(num_ps, cluster_resolver)

      return cluster_resolver, cluster_resolver.cluster_spec()


  def decay(epoch):
      if epoch < 3:
          return 1e-3
      elif epoch >= 3 and epoch < 7:
          return 1e-4
      else:
          return 1e-5

  def get_callbacks(model):
      class PrintLR(tf.keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs=None):
              print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()), flush=True)

      callbacks = [
          tf.keras.callbacks.TensorBoard(log_dir='./logs'),
          tf.keras.callbacks.LearningRateScheduler(decay),
          PrintLR()
      ]
      return callbacks

  def create_dir(dir):
      if not os.path.exists(dir):
          os.makedirs(dir)

  def get_artificial_data():
      x = tf.random.uniform((10, 10))
      y = tf.random.uniform((10,))

      dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
      dataset = dataset.batch(global_batch_size)
      dataset = dataset.prefetch(2)
      return dataset



  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  if not os.environ["OCI__MODE"] == "MAIN":
      create_cluster(cluster_resolver, num_workers=1, num_ps=1, mode=os.environ["OCI__MODE"])
      pass

  variable_partitioner = (
      tf.distribute.experimental.partitioners.MinSizePartitioner(
          min_shard_bytes=(256 << 10),
          max_shards=NUM_PS))

  strategy = tf.distribute.ParameterServerStrategy(
      cluster_resolver,
      variable_partitioner=variable_partitioner)

  dataset = get_artificial_data()

  with strategy.scope():
      model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
      model.compile(tf.keras.optimizers.SGD(), loss="mse", steps_per_execution=10)

  callbacks = get_callbacks(model)
  model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks)





``Train.yaml``: The only difference here is that the parameter server train.yaml also needs to have ``ps`` worker-pool.
This will create dedicated instance(s) for Tensorflow Parameter Servers.

Use the following train.yaml:

.. code-block:: yaml

  kind: distributed
  apiVersion: v1.0
  spec:
    infrastructure:
      kind: infrastructure
      type: dataScienceJob
      apiVersion: v1.0
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: Distributed-TF
        logGroupId: oci.xxxx.<log_group_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.Standard2.4
        blockStorageSize: 50
    cluster:
      kind: TENSORFLOW
      apiVersion: v1.0
      spec:
        image: "@image"
        workDir:  "oci://<bucket_name>@<bucket_namespace>/<bucket_prefix>"
        name: "tf_ps"
        config:
          env:
            - name: WORKER_PORT #Optional. Defaults to 12345
              value: 12345
            - name: SYNC_ARTIFACTS #Mandatory: Switched on by Default.
              value: 1
            - name: WORKSPACE #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket to sync generated artifacts to.
              value: "<bucket_name>"
            - name: WORKSPACE_PREFIX #Mandatory if SYNC_ARTIFACTS==1: Destination object bucket folder to sync generated artifacts to.
              value: "<bucket_prefix>"
        main:
          name: "coordinator"
          replicas: 1 #this will be always 1.
        worker:
          name: "worker"
          replicas: 1 #number of workers; any number > 0
        ps:
          name: "ps" # number of parameter servers; any number > 0
          replicas: 1
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
      spec:
        entryPoint: "/code/train.py" #location of user's training script in the container image.
        args:  #any arguments that the training script requires.
        env:

The rest of the steps remain the same and should be followed as it is.



