===========================
Distributed Training |beta|
===========================

.. |beta| image:: /_static/badge_beta.svg

.. admonition:: Distributed Training with OCI Data Science

  This documentation shows you how to preprocess, and train on a machine
  learning model, using Oracle Cloud Infrastructure. This section will not teach you about distributed training, 
  instead it will help you run your existing distributed training code on OCI Data Science. 



Distributed training is the process of taking a training workload which
comprises training code and training data and making both of these available
in a cluster.

The conceptual difference with distributed training is that multiple workers
coordinated in a cluster running on multiple VM instances allows
horizontal scaling of parallelizable tasks. While singe node training is
well suited to traditional ML models, very large datasets or compute intensive workloads
like deep learning and deep neural networks, tends to be better suited to
distributed computing environments.



Distributed Training benefits two classes of problem, one where the data is parallelizable,
the other where the model network is parallelizable. The most common and easiest to develop
is data parallelism. Both forms of parallelism can be combined to handle both large models
and large datasets.

**Data Parallelism**

In this form of distributed training the training data is partitioned into some multiple
of the number of nodes in the compute cluster. Each node holds the model and is in
communication with other node participating in a coordinated optimization effort.

Sometimes data sampling is possible, but often at the expense of model accuracy. With
distributed training you can avoid having to sample the data to fit a single node.

**Model Parallelism**

This form of distributed training is used when workers need to worker nodes need to synchronize and
share parameters. The data fits into the memory of each worker, but the training takes too long. With
model parallelism more epochs can run and more hyper-parameters can be explored.

**Distributed Training with OCI Data Science**

To outline the process by which you create distributed training workloads is the same regardless of
framework used. Sections of the configuration differ between frameworks but the experience is
consistent. The user brings only the (framework specific) training python code, along with the
``yaml`` declarative definition.

ADS makes use of ``yaml`` to express configurations. The ``yaml`` specification has sections
to describe the cluster infrastructure, the python runtime code, and the cluster framework.

The architecture is extensible to support well known frameworks and future versions of these. The set
of service provided frameworks for distributed training include:

- `Dask <https://docs.dask.org/>`_ for ``LightGBM``, ``XGBoost``, ``Scikit-Learn``,
  and ``Dask-ML``
- `Horovod <https://horovod.ai/>`_ for ``PyTorch`` & ``Tensorflow``
- `PyTorch Distributed <https://pytorch.org/tutorials/beginner/dist_overview.html>`_ for ``PyTorch``
  native using ``DistributedDataParallel`` - no training code changes
  to run PyTorch model training on a cluster. You can use ``Horovod`` to do the same, which has some
  advanced features like auto-tuning to improve
  ``allreduce`` performance, and ``fp16`` gradient compression.
- `Tensorflow Distributed <https://www.tensorflow.org/guide/distributed_training>`_ for ``Tensorflow``
  distributed training strategies like ``MirroredStrategy``, ``MultiWorkerMirroredStrategy`` and
  ``ParameterServerStrategy``


.. toctree::
  :hidden:
  :maxdepth: 5

  cli/distributed-training
  configuration/configuration
  developer/developer
  dask/dask
  horovod/horovod
  pytorch/pytorch
  tensorflow/tensorflow
  remote_source_code
  yaml_schema
  troubleshooting
