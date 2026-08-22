LlamaIndex Integration
**********************

.. versionadded:: 2.12.12

The integrations for LlamaIndex described here allow you to invoke LLMs deployed on OCI Data Science model deployments. With `AI Quick Actions <https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm>`_, deploying and serving large language models becomes straightforward.

For comprehensive guidance on deploying LLM models in OCI Data Science using AI Quick Actions, refer to the detailed documentation available `here <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md>`_ and `here <https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm>`_.


.. admonition:: Requirements
   :class: note

   The LlamaIndex LLM integration requires ``python>=3.9`` and a suitable LLM endpoint deployed on OCI Data Science.
   Ensure you have installed ``llama-index-llms-oci-data-science``, ``llama-index-core`` and ``oracle-ads`` libraries.

.. code-block:: bash

    pip install oracle-ads llama-index-llms-oci-data-science


Authentication
==============

The authentication methods supported for LlamaIndex are equivalent to those used with other OCI services and follow standard SDK authentication methods, including API Key, session token, instance principal, and resource principal. More details can be found `here <https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html>`_. Ensure you have the required `policies <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm>`_ to access the OCI Data Science Model Deployment endpoint.
By default, the integration uses the same authentication method configured with ``ads.set_auth()``. Optionally, you can also pass the ``auth`` keyword argument when initializing the model to use specific authentication method for the model.

Basic Usage
===========

Using LLMs offered by OCI Data Science AI with LlamaIndex only requires you to initialize the ``OCIDataScience`` interface with your Data Science Model Deployment endpoint and model ID. By default, all deployed models in AI Quick Actions get ``odsc-model`` as an ID, but this can be changed during deployment.

Call ``complete`` with a prompt
-----------------------------

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience

   ads.set_auth(auth="resource_principal")

   llm = OCIDataScience(
       # auth=ads.auth.security_token(profile="<replace-with-your-profile>"),
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
       temperature=0.2,
        max_tokens=500,
        timeout=120,
   )
   response = llm.complete("Tell me a joke")

   print(response)

Call ``chat`` with a list of messages
-----------------------------------

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.base.llms.types import ChatMessage

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
   )
   response = llm.chat(
       [
           ChatMessage(role="user", content="Tell me a joke"),
           ChatMessage(role="assistant", content="Why did the chicken cross the road?"),
           ChatMessage(role="user", content="I don't know, why?"),
       ]
   )

   print(response)

Streaming
=========

Using ``stream_complete`` endpoint
-------------------------------
For streaming, a dedicated endpoint must be used: ``/predictWithResponseStream``.

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predictWithResponseStream",
   )

   for chunk in llm.stream_complete("Tell me a joke"):
       print(chunk.delta, end="")

Using ``stream_chat`` endpoint
----------------------------

For streaming, a dedicated endpoint must be used: ``/predictWithResponseStream``.

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.base.llms.types import ChatMessage

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predictWithResponseStream",
   )
   response = llm.stream_chat(
       [
           ChatMessage(role="user", content="Tell me a joke"),
           ChatMessage(role="assistant", content="Why did the chicken cross the road?"),
           ChatMessage(role="user", content="I don't know, why?"),
       ]
   )

   for chunk in response:
       print(chunk.delta, end="")

Async
=====

Call ``acomplete`` with a prompt
------------------------------

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
   )
   response = await llm.acomplete("Tell me a joke")

   print(response)

Call ``achat`` with a list of messages
------------------------------------

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.base.llms.types import ChatMessage

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
   )
   response = await llm.achat(
       [
           ChatMessage(role="user", content="Tell me a joke"),
           ChatMessage(role="assistant", content="Why did the chicken cross the road?"),
           ChatMessage(role="user", content="I don't know, why?"),
       ]
   )

   print(response)

Async Streaming
===============

Using ``astream_complete`` endpoint
---------------------------------

For streaming, a dedicated endpoint must be used: ``/predictWithResponseStream``.

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predictWithResponseStream",
   )

   async for chunk in await llm.astream_complete("Tell me a joke"):
       print(chunk.delta, end="")

Using ``astream_chat`` endpoint
-----------------------------

For streaming, a dedicated endpoint must be used: ``/predictWithResponseStream``.

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.base.llms.types import ChatMessage

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predictWithResponseStream",
   )
   response = await llm.stream_chat(
       [
           ChatMessage(role="user", content="Tell me a joke"),
           ChatMessage(role="assistant", content="Why did the chicken cross the road?"),
           ChatMessage(role="user", content="I don't know, why?"),
       ]
   )

   async for chunk in response:
       print(chunk.delta, end="")

Configure Model
===============

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
       temperature=0.2,
       max_tokens=500,
       timeout=120,
       context_window=2500,
       additional_kwargs={
           "top_p": 0.75,
           "logprobs": True,
           "top_logprobs": 3,
       },
   )
   response = llm.chat(
       [
           ChatMessage(role="user", content="Tell me a joke"),
       ]
   )
   print(response)

Function Calling
================

If the deployed model supports function calling, integration with LlamaIndex tools through ``predict_and_call`` allows the LLM to decide which tools to call.

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.tools import FunctionTool

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
       temperature=0.2,
       max_tokens=500,
       timeout=120,
       context_window=2500,
       additional_kwargs={
           "top_p": 0.75,
           "logprobs": True,
           "top_logprobs": 3,
       },
   )

   def multiply(a: float, b: float) -> float:
       print(f"---> {a} * {b}")
       return a * b

   def add(a: float, b: float) -> float:
       print(f"---> {a} + {b}")
       return a + b

   def subtract(a: float, b: float) -> float:
       print(f"---> {a} - {b}")
       return a - b

   def divide(a: float, b: float) -> float:
       print(f"---> {a} / {b}")
       return a / b

   multiply_tool = FunctionTool.from_defaults(fn=multiply)
   add_tool = FunctionTool.from_defaults(fn=add)
   sub_tool = FunctionTool.from_defaults(fn=subtract)
   divide_tool = FunctionTool.from_defaults(fn=divide)

   response = llm.predict_and_call(
       [multiply_tool, add_tool, sub_tool, divide_tool],
       user_msg="Calculate the result of `8 + 2`.",
       verbose=True,
   )

   print(response)

Using ``FunctionCallingAgent``
------------------------------

.. code-block:: python3

   import ads
   from llama_index.llms.oci_data_science import OCIDataScience
   from llama_index.core.tools import FunctionTool
   from llama_index.core.agent import FunctionCallingAgent

   ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

   llm = OCIDataScience(
       model="odsc-llm",
       endpoint="https://<MD_OCID>/predict",
       temperature=0.2,
       max_tokens=500,
       timeout=120,
       context_window=2500,
       additional_kwargs={
           "top_p": 0.75,
           "logprobs": True,
           "top_logprobs": 3,
       },
   )

   def multiply(a: float, b: float) -> float:
       print(f"---> {a} * {b}")
       return a * b

   def add(a: float, b: float) -> float:
       print(f"---> {a} + {b}")
       return a + b

   def subtract(a: float, b: float) -> float:
       print(f"---> {a} - {b}")
       return a - b

   def divide(a: float, b: float) -> float:
       print(f"---> {a} / {b}")
       return a / b

   multiply_tool = FunctionTool.from_defaults(fn=multiply)
   add_tool = FunctionTool.from_defaults(fn=add)
   sub_tool = FunctionTool.from_defaults(fn=subtract)
   divide_tool = FunctionTool.from_defaults(fn=divide)

   agent = FunctionCallingAgent.from_tools(
       tools=[multiply_tool, add_tool, sub_tool, divide_tool],
       llm=llm,
       verbose=True,
   )
   response = agent.chat(
       "Calculate the result of `8 + 2 - 6`. Use tools. Return the calculated result."
   )

   print(response)
