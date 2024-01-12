LangChain Integration
*********************

.. versionadded:: 2.9.1

LangChain compatible models/interfaces are needed for LangChain applications to invoke OCI generative AI service or LLMs deployed on OCI data science model deployment service.

.. admonition:: Preview Feature
  :class: note

  While the official integration of OCI and LangChain will be added to the LangChain library, ADS provides a preview version of the integration.
  It it important to note that the APIs of the preview version may change in the future.

Integration with Generative AI
==============================

The `OCI Generative AI service <https://www.oracle.com/artificial-intelligence/generative-ai/large-language-models/>`_ provide text generation, summarization and embedding models.

To use the text generation model as LLM in LangChain:

.. code-block:: python3

    from ads.llm import GenerativeAI

    llm = GenerativeAI(
        compartment_id="<compartment_ocid>",
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        },
    )

    llm.invoke("Translate the following sentence into French:\nHow are you?\n")

Here is an example of using prompt template and OCI generative AI LLM to build a translation app:

.. code-block:: python3

    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
    from ads.llm import GenerativeAI
    
    # Map the input into a dictionary
    map_input = RunnableParallel(text=RunnablePassthrough())
    # Template for the input text.
    template = PromptTemplate.from_template(
        "Translate the text into French.\nText:{text}\nFrench translation: "
    )
    llm = GenerativeAI(
        compartment_id="<compartment_ocid>",
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        },
    )

    # Build the app as a chain
    translation_app = map_input | template | llm

    # Now you have a translation app.
    translation_app.invoke("How are you?")
    # "Comment ça va?"

Similarly, you can use the embedding model:

.. code-block:: python3

    from ads.llm import GenerativeAIEmbeddings

    embed = GenerativeAIEmbeddings(
        compartment_id="<compartment_ocid>",
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        },
    )

    embed.embed_query("How are you?")

Integration with Model Deployment
=================================

If you deploy open-source or your own LLM on OCI model deployment service using `vLLM <https://docs.vllm.ai/en/latest/>`_ or `HuggingFace TGI <https://huggingface.co/docs/text-generation-inference/index>`_ , you can use the ``ModelDeploymentVLLM`` or ``ModelDeploymentTGI`` to integrate your model with LangChain.

.. code-block:: python3

    from ads.llm import ModelDeploymentVLLM

    llm = ModelDeploymentVLLM(
        endpoint="https://<your_model_deployment_endpoint>/predict",
        model="<model_name>"
    )

.. code-block:: python3

    from ads.llm import ModelDeploymentTGI

    llm = ModelDeploymentTGI(
        endpoint="https://<your_model_deployment_endpoint>/predict",
    )

Authentication
==============

By default, the integration uses the same authentication method configured with ``ads.set_auth()``. Optionally, you can also pass the ``auth`` keyword argument when initializing the model to use specific authentication method for the model. For example, to use resource principal for all OCI authentication:

.. code-block:: python3

    import ads
    from ads.llm import GenerativeAI
    
    ads.set_auth(auth="resource_principal")
    
    llm = GenerativeAI(
        compartment_id="<compartment_ocid>",
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        },
    )

Alternatively, you may use specific authentication for the model:

.. code-block:: python3

    import ads
    from ads.llm import GenerativeAI

    llm = GenerativeAI(
        # Use security token authentication for the model
        auth=ads.auth.security_token(profile="my_profile"),
        compartment_id="<compartment_ocid>",
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={
            "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        },
    )
