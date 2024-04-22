.. _Notebook Examples:


Notebook Examples
*****************

Below is a compilation of tutorials focused on understanding and utilizing Feature Stores. You can find the raw notebook files in our `tutorials repository <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/feature_store/tutorials/README.md>`_.

Quick start examples
####################

.. admonition:: Notebook Examples
  :class: note

  .. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Jupyter Notebook
      - Description

    * - `Feature store querying <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_querying.ipynb>`__
      - | * Ingestion of data.
        | * Querying and exploration of data.

    * - `Feature store quickstart <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_quickstart.ipynb>`__
      - | * Ingestion of data.
        | * Querying and exploration of data.

    * - `Schema enforcement and schema evolution <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_schema_evolution.ipynb>`__
      - | * ``Schema evolution`` allows you to easily change a table's current schema to accommodate data that is changing over time.
        | * ``Schema enforcement``, also known as schema validation, is a safeguard in Delta Lake that ensures data quality by rejecting writes to a table that don't match the table's schema.

    * - `Storage of medical records in feature store <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_ehr_data.ipynb>`__
      - | Example to demonstrate storage of medical records in Feature Store.

Big data operations using OCI DataFlow
######################################

.. admonition:: Notebook Examples
  :class: note

  .. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Jupyter Notebook
      - Description

    * - `Big data operations with feature store <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_spark_magic.ipynb>`__
      - | * Ingestion of data using Spark Magic.
        | * Querying and exploration of data using Spark Magic.

Streaming operations using Spark Streaming
##########################################

.. admonition:: Notebook Examples
  :class: note

  .. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Jupyter Notebook
      - Description

    * - `Streaming operations with feature store <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_streaming_data_frame.ipynb>`__
      - | * Ingestion of data using spark streaming.
        | * Modes of ingestion: ``COMPLETE`` and ``APPEND``.

LLM Use cases
#############

.. admonition:: Notebook Examples
  :class: note

  .. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Jupyter Notebook
      - Description

    * - `Embeddings in Feature Store <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_embeddings.ipynb>`__
      - | * ``Embedding feature stores`` are optimized for fast and efficient retrieval of embeddings. This is important because embeddings can be high-dimensional and computationally expensive to calculate. By storing them in a dedicated store, you can avoid the need to recalculate embeddings for the same data repeatedly.

    * - `Synthetic data generation in feature store using OpenAI and FewShotPromptTemplate <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_medical_synthetic_data_openai.ipynb>`__
      - | * ``Synthetic data`` is artificially generated data, rather than data collected from real-world events. It's used to simulate real data without compromising privacy or encountering real-world limitations.

    * - `PII Data redaction, Summarise Content and Translate content using doctran and open AI <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_pii_redaction_and_transformation.ipynb>`__
      - | * One way to think of Doctran is a LLM-powered black box where messy strings go in and nice, clean, labelled strings come out. Another way to think about it is a modular, declarative wrapper over OpenAI's functional calling feature that significantly improves the developer experience.

    * - `OpenAI embeddings in feature store <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/notebook_examples/feature_store_embeddings_openai.ipynb>`__
      - | ``Embedding feature stores`` are optimized for fast and efficient retrieval of embeddings. This is important because embeddings can be high-dimensional and computationally expensive to calculate. By storing them in a dedicated store, you can avoid the need to recalculate embeddings for the same data repeatedly.
