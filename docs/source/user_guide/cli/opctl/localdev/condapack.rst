++++++++++++++++++++++++
Working with Conda packs
++++++++++++++++++++++++

Conda packs provide runtime dependencies and a ``python`` runtime for your code. The conda packs can be built inside an ``OCI Data Science Notebook`` session or you can build it locally on your workstation. ``ads opctl`` cli provides a way to setup a development environment to build and use the conda packs. You can push the conda packs that you build locally to Object Storage and use them in Jobs, Notebooks, Pipelines, or in Model Deployments.

**Prerequisites**

1. Build a local ``OCI Data Science Job`` :doc:`compatible docker image<jobs_container_image>`
2. Connect to Object Storage through the Internet
3. Setup conda pack bucket, namespace, and authentication information using ``ads opctl configure``. Refer to configuration :doc:`instructions<../configure>`.

**Note**

* In this version you cannot directly access the Service provided conda environments from ADS CLI, but you can publish a service provided conda pack from an OCI Data Science Notebook session to your object storage bucket and then use the CLI to access the published version. 

------
create
------

.. code-block:: shell

  ads opctl conda create -n <name> -f <path-to-environment-yaml>

Build conda packs from your workstation using ``ads opctl conda create`` subcommand.

-------
publish
-------

.. code-block:: shell

  ads opctl conda publish -s <slug>

Publish conda pack to the object storage bucket from your laptop or workstation. You can use this conda pack inside ``OCI Data Science Jobs``, ``OCI Data Science Notebooks`` and ``OCI Data Science Model Deployment``


-------
install
-------

Install conda pack using its URI. The conda pack can be used inside the docker image that you built. Use Visual Studio Code that is configured with the conda pack to help you test your code locally before submitting to OCI.

.. code-block:: shell

  ads opctl conda install -u "oci://mybucket@namespace/conda_environment/path/to/my/conda"
