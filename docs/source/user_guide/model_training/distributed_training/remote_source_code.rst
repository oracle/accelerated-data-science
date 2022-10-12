==========================================
Run Source Code from Git or Object Storage
==========================================

.. admonition:: Require ADS >=2.6.3

    Running source code from Git or Object Storage requires ADS 2.6.3 or newer.

    ``python3 -m pip install oracle-ads>=2.6.3 --upgrade``

Instead of adding the training source code to the docker image, you can also fetch the code at runtime from Git repository or object storage.

Git Repository
==============

To fetch code from Git repository, you can update the ``runtime`` section of the yaml to specify ``type`` as ``git`` and add ``uri`` of the Git repository to the ``runtime.spec`` section. For example:

.. code-block:: yaml
  :linenos:

  runtime:
    apiVersion: v1
    kind: python
    type: git
    spec:
      uri: git@github.com:username/repository.git
      branch: develop
      commit: abcdef
      gitSecretId: ocid1.xxxxxx
      entryPoint: "train.py"

The ``spec`` supports the following options:

* ``uri``, the URI of the git repository. This can be ``http`` or ``https`` URI for public repository. For private repository, please use ``ssh`` or ``git@`` URI.
* ``branch``, the Git branch. The default branch (usually ``main``) will be used if this is not specified.
* ``commit``, the Git commit. The latest commit will be used if this is not specified.
* ``gitSecretId``, the OCID of secret from OCI vault, which stores the SSH key for accessing private repository.
* ``entryPoint``, the file path to start the training code. The can be the relative path relative to the root of the git repository. The source code is cloned to the ``/code`` directory. You may also use the absolute path.

To clone the git repository, your subnet needs egress from port 80 for ``http``, 443 for ``https``, or 22 for ``ssh``.

You can config proxy for git clone by setting the corresponding ``ssh_proxy``, ``http_proxy`` or ``https_proxy`` environment variable to the proxy address. If you configured ``https_proxy`` or ``http_proxy``, you also need to add all IP addresses in your subnet to the ``no_proxy`` environment variable since communications between training nodes should not go through proxy.

Object Storage
==============

To fetch code from Object Storage, you can update the ``runtime`` section of the yaml to specify ``type`` as ``remote`` and add ``uri`` of the OCI object storage to the ``runtime.spec`` section. For example:

.. code-block:: yaml
  :linenos:

  runtime:
    apiVersion: v1
    kind: python
    type: remote
    spec:
      uri: oci://bucket@namespace/prefix/to/source_code_dir
      entryPoint: "/code/source_code_dir/train.py"

The ``uri`` can be a single file or a prefix (directory). The ``entryPoint`` is the the file path to start the training code. When using relative path, if ``uri`` is a single file, ``entryPoint`` should be the filename. If ``uri`` is a directory, the ``entryPoint`` should contain the name of the directory like the example above. The source code is cloned to the ``/code`` directory. You may also use the absolute path.