.. _vector_store:

#################################################
Integration with OCI Generative AI and OpenSearch
#################################################

.. versionadded:: 2.9.1

OCI Generative Embedding
========================

The Generative AI Embedding Models convert textual input - ranging from phrases and sentences to entire paragraphs - into a structured format known as embeddings. Each piece of text input is transformed into a numerical array consisting of 1024 distinct numbers. The following pretrained model is available for creating text embeddings:

- embed-english-light-v2.0

To find out the latest supported embedding model, check the `documentation <https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm>`_.

The following code snippet shows how to use the Generative AI Embedding Models:

.. code-block:: python3

    from ads.llm import GenerativeAIEmbeddings
    import ads

    ads.set_auth("resource_principal")

    oci_embedings = GenerativeAIEmbeddings(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

Retrieval QA with OpenSearch
============================

OCI OpenSearch
--------------

OCI Search with OpenSearch is a fully managed service which makes searching vast datasets and getting quick results fast and easy. In large language model world, you can use it as a vector store to store your documents and conduct keyword search or semantic search with help of a text embedding model. For a complete walk through on spinning up a OCI OpenSearch Cluster, see `Search and visualize data using OCI Search Service with OpenSearch <https://docs.oracle.com/en/learn/oci-opensearch/index.html#introduction>`_. 

Semantic Search with OCI OpenSearch
-----------------------------------

With the OCI OpenSearch and OCI Generative Embedding, you can do semantic search by using langchain. The following code snippet shows how to do semantic search with OCI OpenSearch:

.. code-block:: python3

    from langchain.vectorstores import OpenSearchVectorSearch
    import os
    # Saving the credentials as environment variables is not recommended. You should save them in Vault instead in prod.
    os.environ['OCI_OPENSEARCH_USERNAME'] = "username" 
    os.environ['OCI_OPENSEARCH_PASSWORD'] = "password"
    os.environ['OCI_OPENSEARCH_VERIFY_CERTS'] = "False" 

    # specify the index name that you would like to conduct semantic search on.
    INDEX_NAME = "your_index_name" 

    opensearch_vector_search = OpenSearchVectorSearch(
        "https://localhost:9200", # your oci opensearch private endpoint
        embedding_function=oci_embedings,
        index_name=INDEX_NAME,
        engine="lucene",
        http_auth=(os.environ["OCI_OPENSEARCH_USERNAME"], os.environ["OCI_OPENSEARCH_PASSWORD"]),
        verify_certs=os.environ["OCI_OPENSEARCH_VERIFY_CERTS"],
    )
    opensearch_vector_search.similarity_search("your query", k=2, size=2)

Retrieval QA Using OCI OpenSearch as a Retriever
------------------------------------------------

Since the search result usually cannot be directly used to answer a specific question. More practical solution is to send the origiral query along with the searched results to a Large Language model to get a more coherent answer. You can also use OCI OpenSearch as a retriever for retrieval QA. The following code snippet shows how to use OCI OpenSearch as a retriever:

.. code-block:: python3

    from langchain.chains import RetrievalQA
    from ads.llm import GenerativeAI

    ads.set_auth("resource_principal")
    
    oci_llm = GenerativeAI(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    retriever = opensearch_vector_search.as_retriever(search_kwargs={"vector_field": "embeds", 
                                                                    "text_field": "text", 
                                                                    "k": 3, 
                                                                    "size": 3})
    qa = RetrievalQA.from_chain_type(
        llm=oci_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True
        }
    )
    qa.run("your question")

Retrieval QA with FAISS
=======================

FAISS as Vector DB
------------------

A lot of the time, your documents are not that large and you dont have a OCI OpenSearch cluster set up. In that case, you can use ``FAISS`` as your in-memory vector store, which can also do similarty search very efficiently. 

The following code snippet shows how to use ``FAISS`` along with OCI Embedding Model to do semantic search:

.. code-block:: python3

    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS

    loader = TextLoader("your.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    l = len(docs)
    embeddings = []
    for i in range(l // 16 + 1):
        subdocs = [item.page_content for item in docs[i * 16: (i + 1) * 16]]
        embeddings.extend(oci_embedings.embed_documents(subdocs))

    texts = [item.page_content for item in docs]
    text_embedding_pairs = [(text, embed) for text, embed in  zip(texts, embeddings)]
    db = FAISS.from_embeddings(text_embedding_pairs, oci_embedings)
    db.similarity_search("your query", k=2, size=2)

Retrieval QA Using FAISS Vector Store as a Retriever
----------------------------------------------------

Similarly, you can use FAISS Vector Store as a retriever to build a retrieval QA engine using langchain. The following code snippet shows how to use OCI OpenSearch as a retriever:

.. code-block:: python3

    from langchain.chains import RetrievalQA
    from ads.llm import GenerativeAI
    import ads

    ads.set_auth("resource_principal")
    
    oci_llm = GenerativeAI(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=oci_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True
        }
    )
    qa.run("your question")

Deployment of Retrieval QA
==========================

As of version 0.0.346, Langchain does not support serialization of any vector stores. This will be a problem when you want to deploy a Retrieval QA langchain application. To solve this problem, we extended our support of vector stores serialization:

- ``OpenSearchVectorSearch``
- ``FAISS``

OpenSearchVectorSearch Serialization
------------------------------------

langchain does not automatically support serialization of ``OpenSearchVectorSearch``. However, ADS provides a way to serialize ``OpenSearchVectorSearch``. To serialize ``OpenSearchVectorSearch``, you need to use environment variables to store the credentials. The following variables can be passed in through the corresponding environment variables:

- http_auth: (``OCI_OPENSEARCH_USERNAME``, ``OCI_OPENSEARCH_PASSWORD``)
- verify_certs: ``OCI_OPENSEARCH_VERIFY_CERTS``
- ca_certs: ``OCI_OPENSEARCH_CA_CERTS``

The following code snippet shows how to use ``OpenSearchVectorSearch`` with environment variables:

.. code-block:: python3

    from langchain.vectorstores import OpenSearchVectorSearch
    import os

    os.environ['OCI_OPENSEARCH_USERNAME'] = "username"
    os.environ['OCI_OPENSEARCH_PASSWORD'] = "password"
    os.environ['OCI_OPENSEARCH_VERIFY_CERTS'] = "False"

    INDEX_NAME = "your_index_name"
    opensearch_vector_search = OpenSearchVectorSearch(
        "https://localhost:9200",
        embedding_function=oci_embedings,
        index_name=INDEX_NAME,
        engine="lucene",
        http_auth=(os.environ["OCI_OPENSEARCH_USERNAME"], os.environ["OCI_OPENSEARCH_PASSWORD"]),
        verify_certs=os.environ["OCI_OPENSEARCH_VERIFY_CERTS"],
    )

.. admonition:: Deployment
  :class: note

During deployment, it is very important that you remember to pass in those environment variables as well or retrieve them from the Vault in score.py which is recommended and more secure:

.. code-block:: python3

    .deploy(deployment_log_group_id="ocid1.loggroup.####",
            deployment_access_log_id="ocid1.log.####",
            deployment_predict_log_id="ocid1.log.####",
            environment_variables={"OCI_OPENSEARCH_USERNAME":"<oci_opensearch_username>",
                                    "OCI_OPENSEARCH_PASSWORD": "<oci_opensearch_password>",
                                    "OCI_OPENSEARCH_VERIFY_CERTS": "<oci_opensearch_verify_certs>",)

Deployment of Retrieval QA with OpenSearch
------------------------------------------

Here is an example code snippet for deployment of Retrieval QA using OpenSearch as a retriever:

.. code-block:: python3

    from ads.llm import GenerativeAIEmbeddings, GenerativeAI
    from ads.llm.deploy import ChainDeployment
    from langchain.chains import RetrievalQA
    from langchain.vectorstores import OpenSearchVectorSearch
    
    import ads
    import os

    ads.set_auth("resource_principal")

    oci_embedings = GenerativeAIEmbeddings(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    oci_llm = GenerativeAI(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )
    # Saving the credentials as environment variables is not recommended. You should save them in Vault instead in prod.
    os.environ['OCI_OPENSEARCH_USERNAME'] = "username"
    os.environ['OCI_OPENSEARCH_PASSWORD'] = "password"
    os.environ['OCI_OPENSEARCH_VERIFY_CERTS'] = "True" # make sure this is capitalized.
    os.environ['OCI_OPENSEARCH_CA_CERTS'] = "path/to/oci_opensearch_ca.pem"

    INDEX_NAME = "your_index_name"
    opensearch_vector_search = OpenSearchVectorSearch(
        "https://localhost:9200", # your endpoint
        embedding_function=oci_embedings,
        index_name=INDEX_NAME,
        engine="lucene",
        http_auth=(os.environ["OCI_OPENSEARCH_USERNAME"], os.environ["OCI_OPENSEARCH_PASSWORD"]),
        verify_certs=os.environ["OCI_OPENSEARCH_VERIFY_CERTS"],
        ca_certs=os.environ["OCI_OPENSEARCH_CA_CERTS"],
    )
    
    retriever = opensearch_vector_search.as_retriever(search_kwargs={"vector_field": "embeds", 
                                                                    "text_field": "text", 
                                                                    "k": 3, 
                                                                    "size": 3})
    qa = RetrievalQA.from_chain_type(
        llm=oci_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True
        }
    )
    
    model = ChainDeployment(qa)
    model.prepare(force_overwrite=True,
            inference_conda_env="<custom_conda_environment_uri>",
            inference_python_version="<python_version>",
            )

    model.save()
    res = model.verify("your prompt")
    model.deploy(deployment_log_group_id="ocid1.loggroup.####",
            deployment_access_log_id="ocid1.log.####",
            deployment_predict_log_id="ocid1.log.####",
            environment_variables={"OCI_OPENSEARCH_USERNAME":"<oci_opensearch_username>",
                                    "OCI_OPENSEARCH_PASSWORD": "<oci_opensearch_password>",
                                    "OCI_OPENSEARCH_VERIFY_CERTS": "<oci_opensearch_verify_certs>",
                                    "OCI_OPENSEARCH_CA_CERTS": "<oci_opensearch_ca_certs>"},)

    model.predict("your prompt")


Deployment of Retrieval QA with FAISS
-------------------------------------

Here is an example code snippet for deployment of Retrieval QA using FAISS as a retriever:

.. code-block:: python3

    from ads.llm import GenerativeAIEmbeddings, GenerativeAI
    from ads.llm.deploy import ChainDeployment
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    
    import ads

    ads.set_auth("resource_principal")
    oci_embedings = GenerativeAIEmbeddings(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    oci_llm = GenerativeAI(
        compartment_id="ocid1.compartment.####",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    loader = TextLoader("your.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    l = len(docs)
    embeddings = []
    for i in range(l // 16 + 1):
        subdocs = [item.page_content for item in docs[i * 16: (i + 1) * 16]]
        embeddings.extend(oci_embedings.embed_documents(subdocs))

    texts = [item.page_content for item in docs]
    text_embedding_pairs = [(text, embed) for text, embed in zip(texts, embeddings)]
    db = FAISS.from_embeddings(text_embedding_pairs, oci_embedings)

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=oci_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True
        }
    )

    model = ChainDeployment(qa)
    model.prepare(force_overwrite=True,
            inference_conda_env="<custom_conda_environment_uri>",
            inference_python_version="<python_version>",
            )

    model.save()
    res = model.verify("your prompt")
    model.deploy(deployment_log_group_id="ocid1.loggroup.####",
            deployment_access_log_id="ocid1.log.####",
            deployment_predict_log_id="ocid1.log.####")

    model.predict("your prompt")
