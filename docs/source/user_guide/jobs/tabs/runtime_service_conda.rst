.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        # Use slug name for conda environment provided by data science service
        .with_service_conda("tensorflow28_p38_cpu_v1")
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      conda:
        type: service
        slug: tensorflow28_p38_cpu_v1
      scriptPathURI: oci://bucket_name@namespace/path/to/script.py
