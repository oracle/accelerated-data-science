Guardrails
**********

ADS provides LangChain compatible guardrails to utilize open-source models or your own models for LLM content moderation.

HuggingFace Measurement as Guardrail
====================================

The ``HuggingFaceEvaluation`` class is designed to take any HuggingFace compatible evaluation and use it as guardrail metrics.
For example, to use the `toxicity measurement <https://huggingface.co/spaces/evaluate-measurement/toxicity>`_ and block any content that has a toxicity score over 0.2:

.. code-block:: python3

    from ads.llm.guardrails.huggingface import HuggingFaceEvaluation

    # Only allow content with toxicity score less than 0.2
    toxicity = HuggingFaceEvaluation(path="toxicity", threshold=0.2)

By default, it uses the `facebook/roberta-hate-speech-dynabench-r4-target<https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target>`_ model. You may use a custom model by specifying the ``load_args`` and ``compute_args``. For example, to use the ``DaNLP/da-electra-hatespeech-detection`` model:

.. code-block:: python3

    toxicity = HuggingFaceEvaluation(
        path="toxicity",
        load_args={"config_name": "DaNLP/da-electra-hatespeech-detection"},
        compute_args={"toxic_label": "offensive"},
        threshold=0.2
    )

To check text with the guardrail, simply call the ``invoke()`` method:

.. code-block:: python3

    toxicity.invoke("<text to be evaluate>")

By default, an exception will be raised if the metric (toxicity in this case) is over the threshold (``0.2`` in this case). Otherwise the same text is returned. 

You can customize this behavior by setting a customized message to be return instead of raising an exception.

.. code-block:: python3

    toxicity = HuggingFaceEvaluation(
        path="toxicity",
        threshold=0.2,
        raise_exception=False,
        custom_msg="Sorry, but let's discuss something else."
    )

Now whenever the input is blocked when calling ``invoke()``, the custom message will be returned.

If you would like to get the value of the metric, you can set the ``return_metrics`` argument to ``True``:

.. code-block:: python3

    toxicity = HuggingFaceEvaluation(path="toxicity", threshold=0.2, return_metrics=True)

In this case, a dictionary containing the metrics will be return when calling ``invoke()``. For example:

.. code-block:: python3

    toxicity.invoke("Oracle is great.")

will give the following outputs:

.. code-block::

    {
        'output': 'Oracle is great.',
        'metrics': {
            'toxicity': [0.00014583684969693422],
            'passed': [True]
        }
    }

Using Guardrail with LangChain
==============================

The ADS guardrail is compatible with LangChain Expression Language (LCEL).
You can use the guardrail with other LangChain components.
In this section we will show how you can use guardrail with a translation application.
The following is a `chain` to translate English to French:

.. code-block:: python3

    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
    from ads.llm import GenerativeAI
    
    # Template for the input text.
    template = PromptTemplate.from_template("Translate the text into French.\nText:{text}\nFrench translation: ")
    llm = GenerativeAI(compartment_id="<compartment_ocid>")
    # Put the output into a dictionary
    map_output = RunnableParallel(translation=RunnablePassthrough())

    # Build the app as a chain
    translation_chain = template | llm | map_output

    # Now you have a translation app.
    translation_chain.invoke({"text": "How are you?"})
    # {'translation': 'Comment ça va?'}

We can add the toxicity guardrail to moderate the user input:

.. code-block:: python3

    from ads.llm.guardrails import HuggingFaceEvaluation

    # Take the text from the input payload for toxicity evaluation
    text = PromptTemplate.from_template("{text}")
    # Evaluate the toxicity and block toxic text.
    toxicity = HuggingFaceEvaluation(path="toxicity", threshold=0.2)
    # Map the text back to a dictionary for the translation prompt template
    map_text = RunnableParallel(text=RunnablePassthrough())

    guarded_chain = text | toxicity | map_text | template | llm | map_output

The ``guarded_chain`` will only translate inputs that are non-toxic.
An exception will be raised if the toxicity of the input is higher than the threshold.

Guardrail Sequence
==================

The ``GuardrailSequence`` class allows you to do more with guardrail and LangChain. You can convert any LangChain ``RunnableSequence`` in to ``GuardrailSequence`` using the ``from_sequence()`` method. For example, with the ``guarded_chain``:

.. code-block:: python3

    from ads.llm.chain import GuardrailSequence

    guarded_sequence = GuardrailSequence.from_sequence(guarded_chain)

We can invoke the ``GuardrailSequence`` in the same way. The output of invoking the ``GuardrailSequence`` not only include the output of the chain, but also the information when running the chain, including parameters and metrics.

.. code-block:: python3

    output = guarded_sequence.invoke({"text": "Hello"})
    # Access the text output from the chain
    print(output.data)
    # {'translation': 'Bonjour'}
    
The ``info`` property of the ``output`` contains a list of run info corresponding to each component in the chain.
For example, to access the toxicity metrics (which is from the second component in the chain)

.. code-block:: python3

    # Access the metrics of the second component
    output.info[1].metrics
    # {'toxicity': [0.00020703606423921883], 'passed': [True]}

The ``GuardrailSequence`` will also stop running the chain once the content is blocked by the guardrail. By default, the custom message from the guardrail will be returned as the output of the sequence.

LLM may generate a wide range of contents, especially when the temperature is set to a higher value. With ``GuardrailSequence``, you can specify a maximum number of retry if the content generated by the LLM is blocked by the guardrail. For example, the following ``detoxified_chain`` will keep re-running the sequence for at most 10 times, until the output of the LLM has a toxicity score that is lower than the threshold.

.. code-block:: python3

    detoxified_chain = GuardrailSequence.from_sequence(llm | toxicity, max_retry=10)
