Run a Script
************

This section shows how to create a job to run a script.

The :py:class:`~ads.jobs.ScriptRuntime` is designed for you to define job artifacts and configurations supported by OCI
Data Science Jobs natively. It can be used with any script types that is supported by the OCI Data Science Jobs,
including shell scripts and python scripts.

The source code can be a single script, files in a folder or a zip/tar file.

See also: `Preparing Job Artifacts <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-artifact.htm>`_.

Here is an example:

.. include:: ../jobs/tabs/script_runtime.rst

An `example script <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/master/jobs/shell/shell-with-args.sh>`_
is available on `Data Science AI Sample GitHub Repository <https://github.com/oracle-samples/oci-data-science-ai-samples>`_.

Working Directory
=================

The working directory is the parent directory where the job artifacts are decompressed,
for example ``/home/datascience/decompressed_artifact/``.
When the source code is a compressed file/archive (zip/tar) or a folder, you need to also specify the entrypoint
using :py:meth:`~ads.jobs.ScriptRuntime.with_entrypoint`. The path of the entrypoint should be a path relative to
the working directory. Note that this directory cannot be changed when using :py:class:`~ads.jobs.ScriptRuntime`.
See :ref:`Python Runtime Working Directory <runtime_working_dir>` for more details.
