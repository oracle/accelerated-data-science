To create a model deployment:

.. tabs::

  .. code-tab:: python
    :caption: Python

    # Deploy model on container runtime
    deployment.deploy()

  .. code-tab:: bash
    :caption: YAML

    # Use the following command to deploy model
    ads opctl run -f ads-md-deploy-<framework>.yaml
