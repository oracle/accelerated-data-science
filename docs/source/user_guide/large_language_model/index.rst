.. _large_language_model:

####################
Large Language Model
####################

Oracle Cloud Infrastructure (OCI) provides fully managed Infrastructure to work with Large Language Model (LLM). You can train LLM at scale with `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_, and deploy it with `Data Science Model Deployment (Model Deployments) <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm>`_. On top of that, you can build and test your LLM applications with LangChain, then deploy it as your own API using the model deployment. 


Compatibility with LangChain
****************************
ADS is designed to be compatible with LangChain, enabling developers to incorporate various LangChain components seamlessly into their langchain applications. 

Deployment Requirements
-----------------------
For successful deployment of LangChain components within ADS, it is crucial to ensure that each component used in the chain is serializable. This is because ADS requires all components to be serializable in order to deploy them as a single unit.

ADS-Supported Components
------------------------
ADS natively supports serialization of all its components. This ensures that any component developed or integrated within ADS adheres to the serialization standards.

Additional LangChain Component Support
--------------------------------------
ADS extends its serialization support to two specific components from the LangChain vector store. These components are:

- ``OpenSearchVectorSearch``: You can connect to the OCI OpenSearch cluster to perform semantic search along with your embedding model. 

- ``FAISS`` (Facebook AI Similarity Search): If you dont have an OCI OpenSearch cluster, you can use FAISS which is a in-memory vector store to perform semantic search along with your embedding model.



.. admonition:: Installation
  :class: note

  Install ADS and other dependencies for LLM integrations.

  .. code-block:: bash

    $ python3 -m pip install "oracle-ads[llm]"



.. toctree::
    :hidden:
    :maxdepth: 2

    training_llm
    deploy_langchain_application
    retrieval