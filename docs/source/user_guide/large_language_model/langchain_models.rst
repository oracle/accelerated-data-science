LangChain Integration
*********************

.. versionadded:: 2.11.19

.. admonition:: LangChain Community
  :class: note

  While the stable integrations (such as ``OCIModelDeploymentVLLM`` and ``OCIModelDeploymentTGI``) are also available from `LangChain Community <https://python.langchain.com/docs/integrations/llms/oci_model_deployment_endpoint>`_, integrations from ADS may provide additional or experimental features in the latest updates, .

.. admonition:: Requirements
  :class: note

  The LangChain integration requires ``python>=3.9`` and ``LangChain>=0.3``


LangChain compatible models/interfaces are needed for LangChain applications to invoke LLMs deployed on OCI data science model deployment service.

If you deploy LLM on OCI model deployment service using `AI Quick Actions <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md>`_ or `HuggingFace TGI <https://huggingface.co/docs/text-generation-inference/index>`_ , you can use the integration models described in this page to build your application with LangChain.

Authentication
==============

By default, the integration uses the same authentication method configured with ``ads.set_auth()``. Optionally, you can also pass the ``auth`` keyword argument when initializing the model to use specific authentication method for the model. For example, to use resource principal for all OCI authentication:

.. code-block:: python3

    import ads
    from ads.llm import ChatOCIModelDeploymentVLLM
    
    ads.set_auth(auth="resource_principal")
    
    llm = ChatOCIModelDeploymentVLLM(
        model="odsc-llm",
        endpoint= f"https://modeldeployment.oci.customer-oci.com/<OCID>/predict",
        # Optionally you can specify additional keyword arguments for the model, e.g. temperature.
        temperature=0.1,
    )

Alternatively, you may use specific authentication for the model:

.. code-block:: python3

    import ads
    from ads.llm import ChatOCIModelDeploymentVLLM

    llm = ChatOCIModelDeploymentVLLM(
        model="odsc-llm",
        endpoint= f"https://modeldeployment.oci.customer-oci.com/<OCID>/predict",
        # Use security token authentication for the model
        auth=ads.auth.security_token(profile="my_profile"),
        # Optionally you can specify additional keyword arguments for the model, e.g. temperature.
        temperature=0.1,
    )

Completion Models
=================

Completion models takes a text string and input and returns a string with completions. To use completion models, your model should be deployed with the completion endpoint (``/v1/completions``). The following example shows how you can use the ``OCIModelDeploymentVLLM`` class for model deployed with vLLM container. If you deployed the model with TGI container, you can use ``OCIModelDeploymentTGI`` similarly.

.. code-block:: python3

    from ads.llm import OCIModelDeploymentVLLM

    llm = OCIModelDeploymentVLLM(
        model="odsc-llm",
        endpoint= f"https://modeldeployment.oci.customer-oci.com/<OCID>/predict",
        # Optionally you can specify additional keyword arguments for the model.
        max_tokens=32,
    )

    # Invoke the LLM. The completion will be a string.
    completion = llm.invoke("Who is the first president of United States?")

    # Stream the completion
    for chunk in llm.stream("Who is the first president of United States?"):
        print(chunk, end="", flush=True)

    # Invoke asynchronously
    completion = await llm.ainvoke("Who is the first president of United States?")

    # Stream asynchronously
    async for chunk in llm.astream("Who is the first president of United States?"):
        print(chunk, end="", flush=True)


Chat Models
===========

Chat models takes `chat messages <https://python.langchain.com/docs/concepts/#messages>`_ as inputs and returns additional chat message (usually `AIMessage <https://python.langchain.com/docs/concepts/#aimessage>`_) as output. To use chat models, your models must be deployed with chat completion endpoint (``/v1/chat/completions``). The following example shows how you can use the ``ChatOCIModelDeploymentVLLM`` class for model deployed with vLLM container. If you deployed the model with TGI container, you can use ``ChatOCIModelDeploymentTGI`` similarly.

.. code-block:: python3

    from langchain_core.messages import HumanMessage, SystemMessage
    from ads.llm import ChatOCIModelDeploymentVLLM

    llm = ChatOCIModelDeploymentVLLM(
        model="odsc-llm",
        endpoint= f"https://modeldeployment.oci.customer-oci.com/<OCID>/predict",
        # Optionally you can specify additional keyword arguments for the model.
        max_tokens=32,
    )

    messages = [
        SystemMessage(content="You're a helpful assistant providing concise answers."),
        HumanMessage(content="Who's the first president of United States?"),
    ]

    # Invoke the LLM. The response will be `AIMessage`
    response = llm.invoke(messages)
    # Print the text of the response
    print(response.content)

    # Stream the response. Note that each chunk is an `AIMessageChunk``
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)

    # Invoke asynchronously
    response = await llm.ainvoke(messages)
    print(response.content)

    # Stream asynchronously
    async for chunk in llm.astream(messages):
        print(chunk.content, end="")


Tool Calling
============

The vLLM container support `tool/function calling <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#automatic-function-calling>`_ on some models (e.g. Mistral and Hermes models). To use tool calling, you must customize the "Model deployment configuration" to use ``--enable-auto-tool-choice`` and specify ``--tool-call-parser`` when deploying the model with vLLM container. A customized ``chat_template`` is also needed for tool/function calling to work with vLLM. ADS includes a convenience way to import the example templates provided by vLLM.

.. code-block:: python3

    from ads.llm import ChatOCIModelDeploymentVLLM, ChatTemplates

    llm = ChatOCIModelDeploymentVLLM(
        model="odsc-llm",
        endpoint= f"https://modeldeployment.oci.customer-oci.com/<OCID>/predict",
        # Set tool_choice to "auto" to enable tool/function calling.
        tool_choice="auto",
        # Use the modified mistral template provided by vLLM
        chat_template=ChatTemplates.mistral()
    )

Following is an example of creating an agent with a tool to get current exchange rate:

.. code-block:: python3

    import requests
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    @tool
    def get_exchange_rate(currency:str) -> str:
        """Obtain the current exchange rates of currency in ISO 4217 Three Letter Currency Code"""

        response = requests.get(f"https://open.er-api.com/v6/latest/{currency}")
        return response.json()

    tools = [get_exchange_rate]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    agent_executor.invoke({"input": "what's the currency conversion of USD to Yen"})
