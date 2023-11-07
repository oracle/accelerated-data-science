.. _Notebook Examples:

==================
Notebook Examples
==================

.. admonition:: Notebook Examples
  :class: note

  .. list-table::
    :widths: 50 50 50
    :header-rows: 1

    * - Html Notebook
      - Jupyter Notebook
      - Description

    * - `Feature store quickstart <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_flights.html>`__
      - `Feature store quickstart <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_flights.ipynb>`__
      - | 1. Ingestion of data.
        | 2. Querying and exploration of data.

    * - `Big data operations with feature store <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature-store-big-data-ingestion-and-querying.html>`__
      - `Big data operations with feature store <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature-store-big-data-ingestion-and-querying.ipynb>`__
      - | 1. Ingestion of data using Spark Magic.
        | 2. Querying and exploration of data using Spark Magic.

    * - `Schema enforcement and schema evolution <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_flights_schema_evolution.html>`__
      - `Schema enforcement and schema evolution <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_flights_schema_evolution.ipynb>`__
      - | 1. Schema evolution allows you to easily change a table's current schema to accommodate data that is changing over time.
        | 2. Schema enforcement, also known as schema validation, is a safeguard in Delta Lake that ensures data quality by rejecting writes to a table that don't match the table's schema.

    * - `Embeddings in Feature Store <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_embeddings.html>`__
      - `Embeddings in Feature Store <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/notebook/feature_store_embeddings.ipynb>`__
      - | 1. One of the primary functions of an embedding feature store is to store pre-trained word embeddings, such as Word2Vec, GloVe, FastText, or BERT embeddings. These embeddings are learned from massive text and contain information about word semantics. Embeddings can be valuable for various NLP tasks like text classification, named entity recognition, sentiment analysis, and so on.
        | 2. Embedding feature stores are optimized for fast and efficient retrieval of embeddings. This is important because embeddings can be high-dimensional and computationally expensive to calculate. By storing them in a dedicated store, you can avoid the need to recalculate embeddings for the same data repeatedly.
