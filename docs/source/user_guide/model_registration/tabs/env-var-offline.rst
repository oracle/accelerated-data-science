Set custom environment variables:

.. tabs::

  .. code-tab:: Python3
    :caption: 7b llama2 - vllm

    env_var = {
        "PARAMS": "--model /opt/ds/model/deployed_model",
    }

  .. code-tab:: Python3
    :caption: 13b llama2 - vllm

    env_var = {
        "PARAMS": "--model /opt/ds/model/deployed_model",
        "TENSOR_PARALLELISM": 2,
    }

  .. code-tab:: Python3
    :caption: 7b llama2 - TGI

    env_var = {
      "MODEL_DEPLOY_PREDICT_ENDPOINT": "/generate",
      "PARAMS": "--model /opt/ds/model/deployed_model --max-batch-prefill-tokens 1024"
    }

  .. code-tab:: Python3
    :caption: 13b llama2 - TGI

    env_var = {
      "MODEL_DEPLOY_PREDICT_ENDPOINT": "/generate",
      "PARAMS" : "--model /opt/ds/model/deployed_model --max-batch-prefill-tokens 1024 --quantize bitsandbytes --max-batch-total-tokens 4096"
    }


You can override more vllm/TGI bootstrapping configuration using ``PARAMS`` environment configuration.
For details of configurations, please refer the official `vLLM doc <https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html>`_ and
`TGI doc <https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_cli>`_.
