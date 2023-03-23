.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_custom_conda("oci://bucket@namespace/conda_pack/pack_name")
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      conda:
        type: published
        uri: oci://bucket@namespace/conda_pack/pack_name
      scriptPathURI: oci://bucket_name@namespace/path/to/script.py
