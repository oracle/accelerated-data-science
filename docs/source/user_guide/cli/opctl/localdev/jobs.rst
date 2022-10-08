+++++++++++++++++++++++++++++++
Build Your Own Container (BYOC)
+++++++++++++++++++++++++++++++



Test Container image
--------------------

OCI Data Science Jobs allows you to use custom container images. ``ads`` cli can help you test a container image locally, publish it, and run it in OCI with a uniform interface.

Running an image locally can be conveniently achieved with "docker run" directly. "ads opctl" commands are provided here only to be symmetric to remote runs on OCI ML Job. The command looks like

.. code-block:: shell

    ads opctl run -i <image-name> -e <docker entrypoint> -c "docker cmd" --env-var ENV_NAME=value -b <backend>

``-b`` option can take either ``local`` - runs the container locally or ``job`` - runs the container on OCI. 


Setup VS Code to use container as development environment
---------------------------------------------------------

During the course of development, it is more productive to work within the container environment to iterate over the code. You can setup your VS Code environment to use the container as your development environment as shown here - 

.. code-block:: shell

    ads opctl init-vscode -i ubuntu --env-var TEST=test -v /Users/<username>/.oci:/root/.oci

A `devcontainer.json` is created with following contents - 

.. code-block:: json

    {
        "image": "ubuntu",
        "mounts": [
            "source=/Users/<username>/.oci,target=/root/.oci,type=bind"
        ],
        "extensions": [
            "ms-python.python"
        ],
        "containerEnv": {
            "TEST": "test"
        }
    }

Publish image to registry
-------------------------

To run a container image with OCI Data Science Job, the image needs to be in a registry accessible by OCI Data Science Job. "ads opctl publish-image" is a thin wrapper on "docker push". The command looks like


.. code-block:: shell

    ads opctl publish-image <image-name>

The image will be pushed to the ``docker registry`` specified in ``ml_job_config.ini``. Check `confiuration <cli/configure>`_ for defaults. To overwrite the registry, use `-r <registry>`.


Run container image on OCI Data Science
---------------------------------------

To run a container on OCI Data Science, provide ``ml_job`` for ``-b`` option. Here is an example - 


.. code-block:: shell

    ads opctl run -i <region>.ocir.io/<tenancy>/ubuntu  -e bash -c '-c "echo $TEST"' -b job --env-var TEST=test




