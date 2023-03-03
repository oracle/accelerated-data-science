.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_environment_variable(
            HOST="10.0.0.1",
            PORT="443",
            URL="http://${HOST}:${PORT}/path/",
            ESCAPED_URL="http://$${HOST}:$${PORT}/path/",
            MISSING_VAR="This is ${UNDEFINED}",
            VAR_WITH_DOLLAR="$10",
            DOUBLE_DOLLAR="$$10"
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      env:
      - name: HOST
        value: 10.0.0.1
      - name: PORT
        value: '443'
      - name: URL
        value: http://${HOST}:${PORT}/path/
      - name: ESCAPED_URL
        value: http://$${HOST}:$${PORT}/path/
      - name: MISSING_VAR
        value: This is ${UNDEFINED}
      - name: VAR_WITH_DOLLAR
        value: $10
      - name: DOUBLE_DOLLAR
        value: $$10
