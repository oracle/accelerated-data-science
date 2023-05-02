.. tabs::

  .. code-tab:: python
    :caption: Python

    runtime = (
      GitPythonRuntime()
      .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
      # Specify the service conda environment by slug name.
      .with_service_conda("pytorch19_p37_gpu_v1")
      # Specify the git repository
      .with_source("https://example.com/your_repository.git")
      # Entrypoint is a relative path from the root of the git repo.
      .with_entrypoint("my_source/my_module.py", func="my_function")
      .with_argument("arg1", "arg2", key1="val1", key2="val2")
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: gitPython
    spec:
      args:
      - arg1
      - arg2
      - --key1
      - val1
      - --key2
      - val2
      conda:
        slug: pytorch19_p37_gpu_v1
        type: service
      entryFunction: my_function
      entrypoint: my_source/my_module.py
      env:
      - name: GREETINGS
        value: Welcome to OCI Data Science
      url: https://example.com/your_repository.git
