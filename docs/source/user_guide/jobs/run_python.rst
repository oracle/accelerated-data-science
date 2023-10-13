Run a Python Workload
*********************

The :py:class:`~ads.jobs.PythonRuntime` is designed for running a Python workload.
You can configure the environment variables, command line arguments, and conda environment
as described in :doc:`infra_and_runtime`. This section shows the additional enhancements provided by
:py:class:`~ads.jobs.PythonRuntime`.


Example
=======

Here is an example to define and run a job using :py:class:`~ads.jobs.PythonRuntime`:

.. include:: ../jobs/tabs/python_runtime.rst

The :py:class:`~ads.jobs.PythonRuntime` uses an driver script from ADS for the job run.
It performs additional operations before and after invoking your code.
You can examine the driver script by downloading the job artifact from the OCI Console.

Source Code
===========

In the :py:meth:`~ads.jobs.PythonRuntime.with_source` method, you can specify the location of your source code.
The location can be a local path or a remote URI supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_.
For example, you can specify files on OCI object storage using URI like
``oci://bucket@namespace/path/to/prefix``. ADS will use the authentication method configured by
:py:meth:`ads.set_auth()` to fetch the files and upload them as job artifact.

The source code can be a single file, a compressed file/archive (zip/tar), or a folder.
When the source code is a compressed file/archive (zip/tar) or a folder, you need to also specify the entrypoint
using :py:meth:`~ads.jobs.PythonRuntime.with_entrypoint`. The path of the entrypoint should be a path relative to
the working directory.

The entrypoint can be a Python script or a Jupyter Notebook.

.. _runtime_working_dir:

Working Directory
=================

The working directory of your workload can be configured by :py:meth:`~ads.jobs.PythonRuntime.with_working_dir`.
By default, :py:class:`~ads.jobs.PythonRuntime` will create a ``code`` directory as the working directory
in the job run to store your source code (job artifact),
for example ``/home/datascience/decompressed_artifact/code``.

When the entrypoint is a Jupyter notebook,
the working directory for the code running in the notebook will be the directory containing the notebook.

When the entrypoint is not a notebook, the working directory depends on the source code.

File Source Code
----------------

If your source code is a single file, for example, ``my_script.py``, the file structure in the job run will look like:
  
.. code-block:: text

  code  <---This is the working directory
  └── my_script.py

You can refer your as ``./my_script.py``

Folder Source Code
------------------

If your source code is a folder, for example ``my_source_code``, ADS will compress the folder as job artifact.
In the job run, it will be decompressed under the working directory. The file structure in the job run will look like:

.. code-block:: text

  code  <---This is the working directory
  └── my_source_code
      ├── my_module.py
      └── my_entrypoint.py

In this case, the working directory is the parent of your source code folder.
You will need to specify the entrypoint as ``my_source_code/my_entrypoint.py``.

.. code-block:: python

  runtime = (
    PythonRuntime()
    .with_source("path/to/my_source_code")
    .with_entrypoint("my_source_code/my_entrypoint.py")
  )

Alternatively, you can specify the working directory as ``my_source_code`` and the entrypoint as ``my_entrypoint.py``:

.. code-block:: python

  runtime = (
    PythonRuntime()
    .with_source("path/to/my_source_code")
    .with_working_dir("my_source_code")
    .with_entrypoint("my_entrypoint.py")
  )

Archive Source Code
-------------------

If your source code is a zip/tar file, the files in the archive will be decompressed under the working directory.
The file structure in the job run depends on whether your archive has a top level directory.
For example, you can inspect the structure of your zip file by running the ``unzip -l`` command:

.. code-block:: bash

  unzip -l my_source_code.zip

This will give you outputs similar to the following:

.. code-block:: text

  Archive:  path/to/my_source_code.zip
    Length      Date    Time    Name
  ---------  ---------- -----   ----
          0  02-22-2023 16:38   my_source_code/
       1803  02-22-2023 16:38   my_source_code/my_module.py
         91  02-22-2023 16:38   my_source_code/my_entrypoint.py
  ---------                     -------
       1894                     3 files

In this case, a top level directory ``my_source_code/`` is presented in the archive.
The file structure in the job run will look like:

.. code-block:: text

  code  <---This is the working directory
  └── my_source_code
      ├── my_module.py
      └── my_entrypoint.py

which is the same as the case when you specified a local folder as source code.
You can configure the entrypoint and working directory similar to the examples above.

If a top level directory is not presented, outputs for the archive will look like the following:

.. code-block:: text

  Archive:  path/to/my_source_code.zip
    Length      Date    Time    Name
  ---------  ---------- -----   ----
       1803  02-22-2023 16:38   my_module.py
         91  02-22-2023 16:38   my_entrypoint.py
  ---------                     -------
       1894                     2 files

In this case, the file structure in the job run will look like:

.. code-block:: text

  code  <---This is the working directory
  ├── my_module.py
  └── my_entrypoint.py

And, you can specify the entrypoint with the filename directly:

.. code-block:: python

  runtime = (
    PythonRuntime()
    .with_source("path/to/my_source_code.zip")
    .with_entrypoint("my_entrypoint.py")
  )

Python Paths
============

The working directory is added to the Python paths automatically.
You can call :py:meth:`~ads.jobs.PythonRuntime.with_python_path` to add additional python paths as needed.
The paths should be relative paths from the working directory.

.. _runtime_outputs:

Outputs
=======

The :py:meth:`~ads.jobs.PythonRuntime.with_output` method allows you to specify the output path ``output_path``
in the job run and a remote URI (``output_uri``).
Files in the ``output_path`` are copied to the remote output URI after the job run finishes successfully.
Note that the ``output_path`` should be a path relative to the working directory.

OCI object storage location can be specified in the format of ``oci://bucket_name@namespace/path/to/dir``.
Please make sure you configure the I AM policy to allow the job run dynamic group to use object storage.
