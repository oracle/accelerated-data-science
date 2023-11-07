Set custom environment variables:

.. tabs::

  .. code-tab:: Python3
    :caption: 7b llama2 - vllm

    env_var = {
        "TOKEN_FILE": "/opt/ds/model/deployed_model/token",
        "PARAMS": "--model meta-llama/Llama-2-7b-chat-hf",
    }

  .. code-tab:: Python3
    :caption: 7b llama2 - TGI

    env_var = {
      "TOKEN_FILE": "/opt/ds/model/deployed_model/token",
      "PARAMS": "--model-id meta-llama/Llama-2-7b-chat-hf --max-batch-prefill-tokens 1024",
    }

  .. code-tab:: Python3
    :caption: 13b llama2 - TGI

    env_var = {
      "TOKEN_FILE": "/opt/ds/model/deployed_model/token",
      "PARAMS" : "--model meta-llama/Llama-2-13b-chat-hf --max-batch-prefill-tokens 1024 --quantize bitsandbytes --max-batch-total-tokens 4096"
    }


You can override more vllm/TGI bootstrapping configuration using ``PARAMS`` environment configuration.
For details of configurations, please refer the official `vLLM doc <https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html>`_ and
`TGI doc <https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_cli>`_.
