AutoGen Integration
*******************

ADS provides custom LLM clients for `AutoGen <https://microsoft.github.io/autogen/0.2/>`_. This custom client allows you to use LangChain chat models for AutoGen.

.. admonition:: Requirements
  :class: note

  The LangChain integration requires ``python>=3.9``, ``langchain-community>=0.3`` and ``langchain-openai``.

  .. code-block:: bash

    pip install "langchain-community>0.3" langchain-openai


Custom Client Registration
==========================

AutoGen requires custom clients to be registered with each agent after the agent is created. To simplify the process, ADS provides a global ``register_model_client()`` method to register the client globally. Once registered with ADS, all new agents created subsequently will have the custom client registered automatically.

The following code shows how you can import the custom client and register it with AutoGen. 

.. code-block:: python3

    from ads.llm.autogen.client_v02 import LangChainModelClient, register_custom_client

    # Register the custom LLM globally
    register_custom_client(LangChainModelClient)

If you don't want the custom client to be registered for all agents. You may skip the above code and still use the ``register_model_client()`` method from each agent.


LLM Config
==========

The LLM config for the ``LangChainModelClient`` should have the following keys:

* ``model_client_cls``, the name of the client class, which should always be ``LangChainModelClient``.
* ``langchain_cls``, the LangChain chat model class with the full path.
* ``model``, the model name for AutoGen to identify the model.
* ``client_params``, the parameters for initializing the LangChain client.

The following keys are optional:
* ``invoke_params``, the parameters for invoking the chat model.
* ``function_call_params``, the parameters for invoking the chat model with functions/tools.

Data Science Model Deployment
-----------------------------

Following is an example LLM config for LLM deployed with AI Quick Action on OCI Data Science Model Deployment:

.. code-block:: python3

    import ads
    from ads.llm.chat_template import ChatTemplates

    # You may use ADS to config the authentication globally
    ads.set_auth("security_token", profile="DEFAULT")

    {
        "model_client_cls": "LangChainModelClient",
        "langchain_cls": "ads.llm.ChatOCIModelDeploymentVLLM",
        # Note that you may use a different model name for the `model` in `client_params`.
        "model": "Mistral-7B",
        # client_params will be used to initialize the LangChain ChatOCIModelDeploymentVLLM class.
        "client_params": {
            "model": "odsc-llm"
            "endpoint": "<ODSC_ENDPOINT>",
            "model_kwargs": {
                "temperature": 0,
                "max_tokens": 500
            },
        }
        # function_call_params will only be added to the API call when function/tools are added.
        "function_call_params": {
            "tool_choice": "auto",
            "chat_template": ChatTemplates.hermes()
        }
    }


OCI Generative AI
-----------------

Following is an example LLM config for the OCI Generative AI service:

.. code-block:: python3

    {
        "model_client_cls": "LangChainModelClient",
        "langchain_cls": "langchain_community.chat_models.oci_generative_ai.ChatOCIGenAI",
        "model": "cohere.command-r-plus",
        # client_params will be used to initialize the LangChain ChatOCIGenAI class.
        "client_params": {
            "model_id": "cohere.command-r-plus",
            "compartment_id": COMPARTMENT_OCID,
            "model_kwargs": {
                "temperature": 0,
                "max_tokens": 4000
            },
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
            "auth_type": "SECURITY_TOKEN",
            "auth_profile": "DEFAULT",
        },
    }

