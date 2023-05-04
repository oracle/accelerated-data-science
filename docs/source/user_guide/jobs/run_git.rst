Run Code from Git Repo
**********************

The :py:class:`~ads.jobs.GitPythonRuntime` allows you to run source code from a Git repository as a job.

.. include:: ../jobs/toc_local.rst

PyTorch Example
===============

The following example shows how to run a
`PyTorch Neural Network Example to train third order polynomial predicting y=sin(x) 
<https://github.com/pytorch/tutorials/blob/master/beginner_source/examples_nn/polynomial_nn.py>`_.

.. include:: ../jobs/tabs/git_runtime.rst


Git Repository
==============

To configure the :py:class:`~ads.jobs.GitPythonRuntime`, you must specify the source code ``url`` and the entrypoint.
The default branch from the Git repository is used unless you specify a different ``branch`` or ``commit``
in the :py:meth:`~ads.jobs.GitPythonRuntime.with_source` method.

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


Entrypoint
==========

The entrypoint specifies how the source code is invoked.
The :py:meth:`~ads.jobs.GitPythonRuntime.with_entrypoint` supports the following arguments:

* ``path``: Required. The relative path of the script/module from the root of the Git repository.
* ``func``: Optional. The function in the script specified by ``path`` to call.
  If you don't specify it, then the script specified by ``path`` is run as a Python script in a subprocess.

The arguments for the entrypoint can be specified through :py:meth:`~ads.jobs.GitPythonRuntime.with_argument`.
For running a script, the arguments are passed in as command line arguments.
See :ref:`Runtime Command Line Arguments <runtime_args>` for more details.
For running a function, the arguments are passed into the function call.

The following example shows how you can define a runtime using Python function from a git repository as an entrypoint.
Here ``my_function`` is a function in the ``my_source/my_module.py`` module.

.. include:: ../jobs/tabs/git_runtime_args.rst

The function will be called as ``my_function("arg1", "arg2", key1="val1", key2="val2")``.

The arguments can be strings, ``list`` of strings or ``dict`` containing only strings.

:py:class:`~ads.jobs.GitPythonRuntime` also support Jupyter notebook as entrypoint.
Arguments are not used when the entrypoint is a notebook.


Working Directory
=================

By default, the working directory is the root of the git repository.
This can be configured by can be configured by :py:meth:`~ads.jobs.GitPythonRuntime.with_working_dir`
using a relative path from the root of the Git repository.

Note that the entrypoint should always specified as a relative path from the root of the Git repository,
regardless of the working directory.
The python paths and output directory should be specified relative to the working directory.


Python Paths
============

The working directory is the root of the git repository.
The working directory is added to the Python paths automatically.
You can call :py:meth:`~ads.jobs.GitPythonRuntime.with_python_path` to add additional python paths as needed.
The paths should be relative paths from the working directory.


Outputs
=======

The :py:meth:`~ads.jobs.GitPythonRuntime.with_output` method allows you to specify the output path ``output_dir``
in the job run and a remote URI (``output_uri``).
Files in the ``output_dir`` are copied to the remote output URI after the job run finishes successfully.
Note that the ``output_dir`` should be a path relative to the working directory.

OCI object storage location can be specified in the format of ``oci://bucket_name@namespace/path/to/dir``.
Please make sure you configure the I AM policy to allow the job run dynamic group to use object storage.


Metadata
========
The :py:class:`~ads.jobs.GitPythonRuntime` updates metadata as free-form tags of the job run
after the job run finishes. The following tags are added automatically:

* ``commit``: The Git commit ID.
* ``method``: The entry function or method.
* ``module``: The entry script or module.
* ``outputs``: The prefix of the output files in Object Storage.
* ``repo``: The URL of the Git repository.

The new values overwrite any existing tags.
If you want to skip the metadata update, set ``skip_metadata_update`` to ``True`` when initializing the runtime:

.. code-block:: python

  runtime = GitPythonRuntime(skip_metadata_update=True)
