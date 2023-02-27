Run a Notebook
**************

The :py:class:`~ads.jobs.NotebookRuntime` allows you to run a single Jupyter notebook as a job.

If your notebook needs extra dependencies like custom module or data files, you can use
:py:class:`~ads.jobs.PythonRuntime` or :py:class:`~ads.jobs.GitPythonRuntime` and set your notebook as the entrypoint.

See also:

* :doc:`run_python`
* :doc:`run_git`

TensorFlow Example
==================

The following example shows you how to run an the
`TensorFlow 2 quick start for beginner
<https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb>`_
notebook from the internet and save the results to OCI Object Storage.
The notebook path points to the raw file link from GitHub.
To run the example, ensure that you have internet access to retrieve the notebook:

.. include:: ../jobs/tabs/notebook_runtime.rst

Working Directory
=================

An empty directory in the job run will be created as the working directory for running the notebook.
All relative paths used in the notebook will be base on the working directory.

Download the Outputs
====================

If you specify the output location using :py:meth:`~ads.jobs.NotebookRuntime.with_output`.
All files in the working directory, including the notebook with outputs,
will be saved to output location (``oci://bucket_name@namespace/path/to/dir``) after the job finishes running.
You can download the output by calling the :py:meth:`~ads.jobs.Job.download` method.

Exclude Cells
=============

The :py:class:`~ads.jobs.NotebookRuntime` also allows you to specify tags to exclude cells from being processed
in a job run using :py:meth:`~ads.jobs.NotebookRuntime.with_exclude_tag` method.
For example, you could do exploratory data analysis and visualization in a notebook,
and you may want to exclude the visualization when running the notebook in a job.

To tag cells in a notebook, see
`Adding tags using notebook interfaces <https://jupyterbook.org/content/metadata.html#adding-tags-using-notebook-interfaces>`__.

The :py:meth:`~ads.jobs.NotebookRuntime.with_exclude_tag` take a list of tags as argument
Cells with any matching tags are excluded from the job run.
In the above example, cells with ``ignore`` or ``remove`` are excluded.
