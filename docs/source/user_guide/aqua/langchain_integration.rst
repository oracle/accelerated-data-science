LangChain Integration
*********************

ADS no longer includes the ``ads.llm`` LangChain integration package.
For LangChain applications that use OCI services or Oracle Database, use Oracle's LangChain integrations from `langchain-oracle <https://github.com/oracle/langchain-oracle>`_.

The Oracle LangChain repository provides:

* ``langchain-oci`` for OCI Generative AI and OCI Data Science Model Deployment integrations.
* ``langchain-oracledb`` for Oracle AI Vector Search integrations.

Install the package that matches your use case:

.. code-block:: bash

   pip install langchain-oci
   pip install langchain-oracledb

See the `langchain-oracle repository <https://github.com/oracle/langchain-oracle>`_ for current installation, API, and example documentation.
