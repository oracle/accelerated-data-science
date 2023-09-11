Train PyTorch Models
********************

.. versionadded:: 2.8.8

The :py:class:`~ads.jobs.PyTorchDistributedRuntime` is designed for training PyTorch models, including large language models (LLMs) with multiple GPUs from multiple nodes. If you develop you training code that is compatible with `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_, `DeepSpeed <https://www.deepspeed.ai/>`_, or `Accelerate <https://huggingface.co/docs/accelerate/index>`_, you can run them using OCI Data Science Jobs with zero code change. For multi-node training, ADS will launch multiple job runs, each corresponding to one node.

See `Distributed Data Parallel in PyTorch <https://pytorch.org/tutorials/beginner/ddp_series_intro.html>`_ for a series of tutorials on PyTorch distributed training.

.. admonition:: Prerequisite
  :class: note

  You need oracle-ads\>=2.8.8 to create a job with :py:class:`~ads.jobs.PyTorchDistributedRuntime`.

  You also need to specify a conda environment with PyTorch\>=1.10 and oracle-ads\>=2.6.8 for the job. See the :ref:`Conda Environment <conda_environment>` about specifying the conda environment for a job.

  We recommend using the ``pytorch20_p39_gpu_v1`` service conda environment and add additional packages as needed.

  You need to specify a subnet ID and allow ingress traffic within the subnet.


Torchrun Example
================

Here is an example to train a GPT model using the source code directly from the official PyTorch Examples Github repository. See `Training "Real-World" models with DDP <https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html>`_ tutorial for a walkthrough of the source code.

.. include:: ../jobs/tabs/pytorch_ddp_torchrun.rst

.. include:: ../jobs/tabs/run_job.rst


Source Code
===========

The source code location can be specified as Git repository, local path or remote URI supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_.

You can use the :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_git` method to specify the source code ``url`` on a Git repository. You can optionally specify the ``branch`` or ``commit`` for checking out the source code. 

For a public repository, we recommend the "http://" or "https://" URL.
Authentication may be required for the SSH URL even if the repository is public.

To use a private repository, you must first save an SSH key to
`OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`_ as a secret,
and provide the ``secret_ocid`` when calling :py:meth:`~ads.jobs.GitPythonRuntime.with_source`.
For more information about creating and using secrets,
see `Managing Secret with Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Tasks/managingsecrets.htm>`_.
For repository on GitHub, you could setup the
`GitHub Deploy Key <https://docs.github.com/en/developers/overview/managing-deploy-keys#deploy-keys>`_ as secret.

.. admonition:: Git Version for Private Repository
  :class: note

  Git version of 2.3+ is required to use a private repository.

Alternatively, you can use the :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_source` method to specify the source code as e a local path or a remote URI supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_.
For example, you can specify files on OCI object storage using URI like
``oci://bucket@namespace/path/to/prefix``. ADS will use the authentication method configured by
:py:meth:`ads.set_auth()` to fetch the files and upload them as job artifact. The source code can be a single file, a compressed file/archive (zip/tar), or a folder.

Working Directory
=================

The default working directory depends on how the source code is specified.
* When the source code is specified as Git repository URL, the default working directory is the root of the Git repository.
* When the source code is a single file (script), the default working directory containing the file.
* When the source code is specified as a local or remote directory, the default working directory is the directory containing the source code directory.

The working directory of your workload can be configured by :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_working_dir`. See :ref:`Python Runtime Working Directory <runtime_working_dir>` for more details.

Input Data
==========

You can specify the input (training) data for the job using the :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_inputs` method, which takes a dictionary mapping the "source" to the "destination". The "source" can be an OCI object storage URI, HTTP or FTP URL. The "destination" is the local path in a job run. If the "destination" is specified as relative path, it will be relative to the working directory.

Outputs
=======

You can specify the output data to be copied to the object storage by using the :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_output` method.
It allows you to specify the output path ``output_path``
in the job run and a remote URI (``output_uri``).
Files in the ``output_path`` are copied to the remote output URI after the job run finishes successfully.
Note that the ``output_path`` should be a path relative to the working directory.

OCI object storage location can be specified in the format of ``oci://bucket_name@namespace/path/to/dir``.
Please make sure you configure the I AM policy to allow the job run dynamic group to use object storage.

Number of nodes
===============

The :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_replica` method helps you to specify the number node for the training job.

Command
=======

The command to start your workload is specified by using the :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_command` method.

For ``torchrun``, ADS will set ``--nnode``, ``--nproc_per_node``, ``--rdzv_backend`` and ``--rdzv_endpoint`` automatically. You do not need to specify them in the command unless you would like to override the values. The default ``rdzv_backend`` will be ``c10d``. The default port for ``rdzv_endpoint`` is 29400

If you workload uses Deepspeed, you also need to set ``use_deepspeed`` to ``True`` when specifying the command. For Deepspeed, ADS will generate the hostfile automatically and setup the SSH configurations.

For ``accelerate launch``, you can add your config YAML to the source code and specify it using ``--config_file`` argument. In your config, please use ``LOCAL_MACHINE`` as the compute environment. The same config file will be used by all nodes in multi-node workload. ADS will set ``--num_processes``, ``--num_machines``, ``--machine_rank``, ``--main_process_ip`` and ``--main_process_port`` automatically. For these arguments, ADS will override the values from your config YAML. If you would like to use your own values, you need to specify them as command arguments. The default ``main_process_port`` is 29400.

Additional dependencies
=======================

The :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_dependency` method helps you to specify additional dependencies to be installed into the conda environment before starting your workload.
* ``pip_req`` specifies the path of the ``requirements.txt`` file in your source code.
* ``pip_pkg`` specifies the packages to be installed as a string.

Python Paths
============

The working directory is added to the Python paths automatically.
You can call :py:meth:`~ads.jobs.PyTorchDistributedRuntime.with_python_path` to add additional python paths as needed.
The paths should be relative paths from the working directory.

