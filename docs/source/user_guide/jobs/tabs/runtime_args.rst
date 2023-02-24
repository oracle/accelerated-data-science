.. tabs::

  .. code-tab:: python
    :caption: Python

    runtime = PythonRuntime()
    runtime.with_argument(key1="val1", key2="val2")
    runtime.with_argument("pos1")

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      args:
      - --key1
      - val1
      - --key2
      - val2
      - pos1

.. code-block:: python

  print(runtime.args)
  # ['--key1', 'val1', '--key2', 'val2', 'pos1']

.. tabs::

  .. code-tab:: python
    :caption: Python

    runtime = PythonRuntime()
    runtime.with_argument("pos1")
    runtime.with_argument(key1="val1", key2="val2.1 val2.2")
    runtime.with_argument("pos2")

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      args:
      - pos1
      - --key1
      - val1
      - --key2
      - val2.1 val2.2
      - pos2

.. code-block:: python

  print(runtime.args)
  # ['pos1', '--key1', 'val1', '--key2', 'val2.1 val2.2', 'pos2']

.. tabs::

  .. code-tab:: python
    :caption: Python

    runtime = PythonRuntime()
    runtime.with_argument("pos1")
    runtime.with_argument(key1=None, key2="val2")
    runtime.with_argument("pos2")

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      args:
      - pos1
      - --key1
      - --key2
      - val2
      - pos2

.. code-block:: python

  print(runtime.args)
  # ["pos1", "--key1", "--key2", "val2", "pos2"]