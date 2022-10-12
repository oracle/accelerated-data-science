+++++++++++++++++++++++++++++
Setting up Visual Studio Code
+++++++++++++++++++++++++++++

`Visual Studio Code <https://code.visualstudio.com/>`_ can automatically run the code that you are developing inside a preconfigured container. An ``OCI Data Science`` compatible container on your workstation can be used as a development environment. Visual Studio Code can automatically launch the container using the information from ``devcontainer.json``, which is created in the code directory. Automatically generate this file and further customize it with plugins. For more details `see <https://code.visualstudio.com/docs/remote/devcontainerjson-reference>`_

**Prerequisites**

1. ADS CLI is :doc:`configured<../configure>`
2. Install Visual Studio Code
3. :doc:`Build Development Container Image<jobs_container_image>`
4. Install Visual Studio Code extension for `Remote Development <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_

.. code-block:: shell

  ads opctl init-vscode -s <source-folder>

``source-folder`` is a directory on your workstation where the code will reside.

``env-var`` - Use this option to setup the environment variables required when the container used for development is started.

If you have to setup a proxy, you can use the following command - 

.. code-block:: shell

  ads opctl init-vscode -s <source-folder> --env-var http_proxy=$http_proxy https_proxy=$https_proxy no_proxy=$no_proxy

The generated ``.devcontainer.json`` includes the python extension for Visual Studio Code by default.

Open the ``source_folder`` using Visual Studio Code. More details on running the workspace within the container can be found `here <https://code.visualstudio.com/docs/remote/containers-tutorial>`_
