* :py:class:`~ads.jobs.PythonRuntime`
  for Python code stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_. See :doc:`run_python`.
* :py:class:`~ads.jobs.GitPythonRuntime`
  for Python code from a Git repository. See :doc:`run_git`.
* :py:class:`~ads.jobs.NotebookRuntime`
  for a single Jupyter notebook stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_. See :doc:`run_notebook`.
* :py:class:`~ads.jobs.ScriptRuntime`
  for bash or shell scripts stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_. See :doc:`run_script`.