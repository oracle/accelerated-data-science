Run a Container
***************

The :py:class:`~ads.jobs.ContainerRuntime` class allows you to run a container image using OCI data science jobs.

.. admonition:: OCI Container Registry
  :class: note

  To use the :py:class:`~ads.jobs.ContainerRuntime`, you need to first push the image to
  `OCI container registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_.

  Note that you cannot build a docker image inside an OCI Data Science Notebook Session.

  For more details, see:
  
  * `Creating a Repository <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_
  * `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_.

Here is an example to create and run a container job:

.. include:: ../jobs/tabs/container_runtime.rst

To configure ``ContainerRuntime``, you must specify the container ``image``.
Similar to other runtime, you can add environment variables.
You can optionally specify the `entrypoint`, `cmd`, `image_digest` and `image_signature_id` for running the container. You may also add additional artifact (file or directory) if needed. Please note that if you add a directory, it will be compressed as a zip file under `/home/datascience` and you will need to unzip if in your container.

See also:

* `Understand how CMD and ENTRYPOINT interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_
* `Bring Your Own Container <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-byoc.htm>`_
