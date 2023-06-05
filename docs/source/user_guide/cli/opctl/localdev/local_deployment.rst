++++++++++++++++++++++++++++++++
Local Model Deployment Execution
++++++++++++++++++++++++++++++++

You can test whether deployment will work in a local container to facilitate development and troubleshooting by testing local predict.

-------------
Prerequisites
-------------

1. :doc:`Install ADS CLI<../../quickstart>`
2. Build a container image.
    - :doc:`Build Development Container Image<./jobs_container_image>`

------------
Restrictions
------------

When running locally, your local predict is subject to the following restrictions:
  - The local predict must use API Key auth. Resource Principal auth is not supported in a local container. See https://docs.oracle.com/iaas/Content/API/Concepts/apisigningkey.htm
  - You can only use conda environment published to your own Object Storage bucket. See :doc:`Working with Conda packs<./condapack>`
  - Your model artifact files must be present on your local machine.

--------------------------
Running your Local Predict
--------------------------

.. note:: Right now, we only support testing deployment locally using a conda environment.

Using a conda environment
=========================

This example below demonstrates how to run a local predict using an installed conda environment:

.. code-block:: shell

  ads opctl predict --ocid "ocid1.datasciencemodel.oc1.iad.<ocid>" --payload '[[-1.68671955,2.25814541,-0.5068027,0.25248417,0.62665134,0.23441123]]' --conda-slug myconda_p38_cpu_v1

Parameter explanation:
  - ``--ocid``: Run the predict locally in a docker container when you pass in a model id. If you pass in a deployment id, e.g. ``ocid1.datasciencemodeldeployment.oc1.iad.``, it will actually predict against the remote endpoint. In that case, only ``--ocid`` and ``--payload`` are needed.
  - ``--conda-slug myconda_p38_cpu_v1``:  Use the ``myconda_p38_cpu_v1`` conda environment. The environment should be installed in your local already. If you haven't installed it, you can provide the path by ``--conda-path``, for example, ``--conda-path "oci://my-bucket@mytenancy/.../myconda_p38_cpu_v1"``, it will download and install the conda environment in your local. Note that you must publish this conda environment to you own bucket first if you are actually using a service conda pack for your real deployment. We will find to auto detect the conda information from the custom metadata, then the runtime.yaml from your model artifact if ``--conda-slug`` and ``--conda-path`` is not provided. However, if detected conda are service packs, we will throw an error asking you to publish it first. Note, in order to test whether deployemnt will work, you should provide the slug that you will use in real deployment. The local conda environment directory will be automatically mounted into the container and activated before the entrypoint is executed.
  - ``--payload``: The payload to be passed to your model.
  - ``--bucket-uri``: Used to download large model artifact to your local machine. Extra policy needs to be placed for this bucket. Check this :doc:`link <./user_guide/model_registration/model_load.html#large-model-artifacts>` for more details. Small artifact does not require this.

.. code-block:: shell

  ads opctl predict --artifact-directory /folder/your/model/artifacts/are/stored --payload '[[-1.68671955,2.25814541,-0.5068027,0.25248417,0.62665134,0.23441123]]'

Parameter explanation:
  - ``--artifact-directory /folder/your/model/artifacts/are/stored``: If you already have your model artifact stored in some local folder, you can use ``--artifact-directory`` instead of ``--ocid``.

----------------------------------------------------
Running your Predict Against the Deployment Endpoint
----------------------------------------------------
.. code-block:: shell

  ads opctl predict --ocid "ocid1.datasciencemodeldeployment.oc1.iad.<ocid>" --payload '[[-1.68671955,2.25814541,-0.5068027,0.25248417,0.62665134,0.23441123]]'


Parameter explanation:
  - ``--ocid ocid1.datasciencemodeldeployment.oc1.iad.<ocid>``: Run the predict remotely against the remote endpoint.
  - ``--payload``: The payload to be passed to your model.
