.. _data-labeling-8:

##########
Label Data
##########

The Oracle Cloud Infrastructure (OCI) Data Labeling service allows you to create and browse datasets, view data records (text, images) and apply labels for the purposes of building AI/machine learning (ML) models.  The service also provides interactive user interfaces that enable the labeling process.  After you label records, you can export the dataset as line-delimited JSON Lines (JSONL) for use in  model development.

Datasets are the core resource available within the Data Labeling service. They contain records and their associated labels.  A record represents a single image or text document. Records are stored by reference to their original source such as path on Object Storage. You can also upload records from local storage. Labels are annotations that describe a data record.  There are three different dataset formats, each having its respective annotation classes:

* Images: Single label, multiple label, and object detection. Supported image types are ``.png``, ``.jpeg``, and ``.jpg``.
* Text: Single label, multiple label, and entity extraction. Plain text, ``.txt``, files are supported.
* Document: Single label and multiple label. Supported document types are ``.pdf`` and ``.tiff``.

.. toctree::
   :hidden:
   :maxdepth: 1

   overview
   quick_start
   export
   list
   load
   visualize
   example
