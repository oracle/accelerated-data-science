EmbeddingONNXModel
******************

See `API Documentation <../../../ads.model.framework.html#ads.model.framework.embedding_onnx_model.EmbeddingONNXModel>`__

Overview
========

The ``ads.model.framework.embedding_onnx_model.EmbeddingONNXModel`` class in ADS is designed to rapidly get an Embedding ONNX Model into production. The ``.prepare()`` method creates the model artifacts that are needed without configuring it or writing code. ``EmbeddingONNXModel`` supports `OpenAI spec <https://github.com/huggingface/text-embeddings-inference/blob/main/docs/openapi.json>`_ for embeddings endpoint.

.. include:: ../_template/overview.rst

The following steps take the `sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ model and deploy it into production with a few lines of code.


**Download Embedding Model from HuggingFace**

.. code-block:: python3

    import tempfile
    import os
    import shutil
    from huggingface_hub import snapshot_download

    local_dir = tempfile.mkdtemp()

    allow_patterns=[
        "onnx/model.onnx",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]

    # download files needed for this demostration to local folder
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=local_dir,
        allow_patterns=allow_patterns
    )

    artifact_dir = tempfile.mkdtemp()
    # copy all downloaded files to artifact folder
    for file in allow_patterns:
        shutil.copy(local_dir + "/" + file, artifact_dir)


Install Conda Pack
==================

To deploy the embedding onnx model, start with the onnx conda pack with slug ``onnxruntime_p311_gpu_x86_64``. 

.. code-block:: bash

    odsc conda install -s onnxruntime_p311_gpu_x86_64


Prepare Model Artifact
======================

Instantiate an ``EmbeddingONNXModel()`` object with Embedding ONNX model. All the model related files will be saved under ``artifact_dir``. ADS will auto generate the ``score.py`` and ``runtime.yaml`` that are required for the deployment.

For more detailed information on what parameters that ``EmbeddingONNXModel`` takes, refer to the `API Documentation <../../../ads.model.framework.html#ads.model.framework.embedding_onnx_model.EmbeddingONNXModel>`__


.. code-block:: python3

    import ads
    from ads.model import EmbeddingONNXModel

    # other options are `api_keys` or `security_token` depending on where the code is executed
    ads.set_auth("resource_principal")

    embedding_onnx_model = EmbeddingONNXModel(artifact_dir=artifact_dir)
    embedding_onnx_model.prepare(
        inference_conda_env="onnxruntime_p311_gpu_x86_64",
        inference_python_version="3.11",
        model_file_name="model.onnx",
        force_overwrite=True
    )


Summary Status
==============

.. include:: ../_template/summary_status.rst

.. figure:: ../figures/summary_status.png
   :align: center


Verify Model
============

Call the ``verify()`` to check if the model can be executed locally.

.. code-block:: python3

    embedding_onnx_model.verify(
        {
            "input": ['What are activation functions?', 'What is Deep Learning?'],
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
    )

If successful, similar results as below should be presented.

.. code-block:: python3

    {
        'object': 'list',
        'data': 
            [{
                'object': 'embedding',
                'embedding': 
                    [[
                        -0.11011122167110443,
                        -0.39235609769821167,
                        0.38759472966194153,
                        -0.34653618931770325,
                        ...,
                    ]]
            }]
    }

Register Model
==============

Save the model artifacts and create an model entry in OCI DataScience Model Catalog.

.. code-block:: python3

    embedding_onnx_model.save(display_name="sentence-transformers/all-MiniLM-L6-v2")


Deploy and Generate Endpoint
============================

Create a model deployment from the embedding onnx model in Model Catalog. The process takes several minutes and the deployment configurations will be presented once it's completed.

.. code-block:: python3

    embedding_onnx_model.deploy(
        display_name="all-MiniLM-L6-v2 Embedding Model Deployment",
        deployment_log_group_id="<log_group_id>",
        deployment_access_log_id="<access_log_id>",
        deployment_predict_log_id="<predict_log_id>",
        deployment_instance_shape="VM.Standard.E4.Flex",
        deployment_ocpus=20,
        deployment_memory_in_gbs=256,
    )


Run Prediction against Endpoint
===============================

Call ``predict()`` to check the model deployment endpoint. 

.. code-block:: python3

    embedding_onnx_model.predict(
        {
            "input": ["What are activation functions?", "What is Deep Learning?"],
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
    )

If successful, similar results as below should be presented.

.. code-block:: python3

    {
        'object': 'list',
        'data': 
            [{
                'object': 'embedding',
                'embedding': 
                    [[
                        -0.11011122167110443,
                        -0.39235609769821167,
                        0.38759472966194153,
                        -0.34653618931770325,
                        ...,
                    ]]
            }]
    }

Run Prediction with OCI CLI
===========================

Model deployment endpoints can also be invoked with the OCI CLI.

.. code-block:: bash

    oci raw-request --http-method POST --target-uri <deployment_endpoint> --request-body '{"input": ["What are activation functions?", "What is Deep Learning?"], "model": "sentence-transformers/all-MiniLM-L6-v2"}' --auth resource_principal


Example
=======

.. code-block:: python3

    import tempfile
    import os
    import shutil
    import ads
    from ads.model import EmbeddingONNXModel
    from huggingface_hub import snapshot_download

    # other options are `api_keys` or `security_token` depending on where the code is executed
    ads.set_auth("resource_principal")

    local_dir = tempfile.mkdtemp()

    allow_patterns=[
        "onnx/model.onnx",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]

    # download files needed for this demostration to local folder
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=local_dir,
        allow_patterns=allow_patterns
    )

    artifact_dir = tempfile.mkdtemp()
    # copy all downloaded files to artifact folder
    for file in allow_patterns:
        shutil.copy(local_dir + "/" + file, artifact_dir)

    # initialize EmbeddingONNXModel instance and prepare score.py, runtime.yaml and openapi.json files.
    embedding_onnx_model = EmbeddingONNXModel(artifact_dir=artifact_dir)
    embedding_onnx_model.prepare(
        inference_conda_env="onnxruntime_p311_gpu_x86_64",
        inference_python_version="3.11",
        model_file_name="model.onnx",
        force_overwrite=True
    )

    # validates model locally
    embedding_onnx_model.verify(
        {
            "input": ['What are activation functions?', 'What is Deep Learning?'],
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
    )

    # save model to oci model catalog
    embedding_onnx_model.save(display_name="sentence-transformers/all-MiniLM-L6-v2")

    # deploy model
    embedding_onnx_model.deploy(
        display_name="all-MiniLM-L6-v2 Embedding Model Deployment",
        deployment_log_group_id="<log_group_id>",
        deployment_access_log_id="<access_log_id>",
        deployment_predict_log_id="<predict_log_id>",
        deployment_instance_shape="VM.Standard.E4.Flex",
        deployment_ocpus=20,
        deployment_memory_in_gbs=256,
    )

    # check model deployment endpoint
    embedding_onnx_model.predict(
        {
            "input": ["What are activation functions?", "What is Deep Learning?"],
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
    )
