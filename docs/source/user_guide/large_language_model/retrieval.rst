.. _vector_store:

##########################################################
Extensive Support of Langchain Vector Stores serialization
##########################################################

.. versionadded:: 2.9.1

Current version of Langchain does not support serialization of any vector stores. This will be a problem when you want to deploy a langchain application with the vector store being one of the components using data science model deployment service. To solve this problem, we extended our support of vector stores serialization:

- ``OpenSearchVectorSearch``
- ``FAISS``

OpenSearchVectorSearch Serialization
------------------------------------

langchain does not automatically support serialization of ``OpenSearchVectorSearch``. However, ADS provides a way to serialize ``OpenSearchVectorSearch``. To serialize ``OpenSearchVectorSearch``, you need to use environment variables to pass in the credentials. The following variables can be passed in through the corresponding environment variables:

- http_auth: (``OCI_OPENSEARCH_USERNAME``, ``OCI_OPENSEARCH_PASSWORD``)
- verify_certs: ``OCI_OPENSEARCH_VERIFY_CERTS``
- ca_certs: ``OCI_OPENSEARCH_CA_CERTS``

The following code snippet shows how to use ``OpenSearchVectorSearch`` with environment variables:

.. code-block:: python3

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
  
During deployment, it is very important that you remember to pass in those environment variables as well:

.. code-block:: python3

    .deploy(deployment_log_group_id="ocid1.loggroup.####",
            deployment_access_log_id="ocid1.log.####",
            deployment_predict_log_id="ocid1.log.####",
            environment_variables={"OCI_OPENSEARCH_USERNAME":"<oci_opensearch_username>",
                                    "OCI_OPENSEARCH_PASSWORD": "<oci_opensearch_password>",
                                    "OCI_OPENSEARCH_VERIFY_CERTS": "<oci_opensearch_verify_certs>",)

OpenSearchVectorSearch Deployment
---------------------------------

Here is an example code snippet for OpenSearchVectorSearch deployment:

.. code-block:: python3

    from langchain.vectorstores import OpenSearchVectorSearch
    from ads.llm import GenerativeAIEmbeddings, GenerativeAI
    import ads

    ads.set_auth("resource_principal")

    oci_embedings = GenerativeAIEmbeddings(
        compartment_id="ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    oci_llm = GenerativeAI(
        compartment_id="ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq",
        client_kwargs=dict(service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com") # this can be omitted after Generative AI service is GA.
    )

    import os
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
    from langchain.chains import RetrievalQA
    retriever = opensearch_vector_search.as_retriever(search_kwargs={"vector_field": "embeds", 
                                                                    "text_field": "text", 
                                                                    "k": 3, 
                                                                    "size": 3},
                                                    max_tokens_limit=1000)
    qa = RetrievalQA.from_chain_type(
        llm=oci_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True
        }
    )
    from ads.llm.deploy import ChainDeployment
    model = ChainDeployment(qa)
    model.prepare(force_overwrite=True,
            inference_conda_env="your_conda_pack",
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


FAISS Serialization
-------------------

If your documents are not too large and you dont have a OCI OpenSearch cluster, you can use ``FAISS`` as your in-memory vector store, which can also do similarty search very efficiently. For ``FAISS``, you can just use it and deploy it as it is.


FAISS Deployment
----------------

Here is an example code snippet for FAISS deployment:

.. code-block:: python3

    import ads
    from ads.llm import GenerativeAIEmbeddings, GenerativeAI
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA

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
    text_embedding_pairs = [(text, embed) for text, embed in  zip(texts, embeddings)]
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

    from ads.llm.deploy import ChainDeployment
    model.prepare(force_overwrite=True,
            inference_conda_env="your_conda_pack",
            )

    model.save()
    res = model.verify("your prompt")
    model.deploy(deployment_log_group_id="ocid1.loggroup.####",
            deployment_access_log_id="ocid1.log.####",
            deployment_predict_log_id="ocid1.log.####")

    model.predict("your prompt")
