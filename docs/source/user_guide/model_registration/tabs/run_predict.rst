Run inference against the deployed model:

.. tabs::

  .. code-tab:: python
    :caption: Python

    # For TGI
    data = {
        "inputs": "Write a python program to randomly select item from a predefined list?",
        "parameters": {
          "max_new_tokens": 200
        }
      }

    # For vLLM
    data = {
        "prompt": "are you smart?",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
      }

    deployment.predict(data=data)

  .. code-tab:: bash
    :caption: TGI Inference by OCI CLI

    oci raw-request \
        --http-method POST \
        --target-uri "<TGI_model_endpoint>" \
        --request-body '{
            "inputs": "Write a python program to randomly select item from a predefined list?",
            "parameters": {
            "max_new_tokens": 200
            }
        }' \
        --auth resource_principal

  .. code-tab:: bash
    :caption: vLLM Inference by OCI CLI

    oci raw-request \
      --http-method POST \
      --target-uri "<vLLM_model_endpoint>" \
      --request-body '{
        "prompt": "are you smart?",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
      }' \
      --auth resource_principal
