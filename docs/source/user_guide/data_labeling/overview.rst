Overview
========

The Oracle Cloud Infrastructure (OCI) Data Labeling service allows you to create and browse datasets,
view data records (text, images) and apply labels for the purposes of building AI/machine learning (ML) models.
The service also provides interactive user interfaces that enable the labeling process.
Afert you label records, you can export the dataset as line-delimited JSON Lines (JSONL) for use in  model development.

Datasets are the core resource available within the Data Labeling service. They contain records and their associated labels.
A record represents a single image or text document. Records are stored by reference to their original source such as path on Object Storage. You can also upload records from local storage. Labels are annotations that describe a data record.

There are three different dataset formats, each having its respective annotation classes:

   - Images: Single label, multiple label, and object detection. Supported image types are ``.png``, ``.jpeg``, and ``.jpg``.
   - Text: Single label, multiple label, and entity extraction. Plain text, ``.txt``, files are supported.
   - Document: Single label and multiple label. Supported document types are ``.pdf`` and ``.tiff``.


Quick Start
-----------

The following examples provide an overview of how to use ADS to work with the Data Labeling service.

List all the datasets in the compartment:

.. code:: ipython3

   from ads.data_labeling import DataLabeling
   dls = DataLabeling()
   dls.list_dataset()


With a labeled data set, the details of the labeling is called the export.
To generate the export and get the path to the metadata JSONL file,
you can use ``export()`` with these parameters:

- `dataset_id`: The OCID of the Data Labeling dataset to take a snapshot of.
- `path`: The Object Storage path to store the generated snapshot.

.. code:: ipython3

   metadata_path = dls.export(
       dataset_id="<dataset_id>",
       path="oci://<bucket_name>@<namespace>/<prefix>"
   )


To load the labeled data into a Pandas dataframe, you can use ``LabeledDatasetReader`` object
that has these parameters:

- `materialize`: Load the contents of the dataset. This can be quite large. The default is `False`.
- `path`: The metadata file path that can be local or object storage path.

.. code:: ipython3

   from ads.data_labeling import LabeledDatasetReader
   ds_reader = LabeledDatasetReader.from_export(
     path="<metadata_path>",
     materialize=True
   )
   df = ds_reader.read()

You can also read labeled datasets from the OCI Data Labeling Service into a Pandas dataframe using ``LabeledDatasetReader`` object by specifying
``dataset_id``:


.. code:: ipython3

   from ads.data_labeling import LabeledDatasetReader
   ds_reader = LabeledDatasetReader.from_DLS(
     dataset_id="<dataset_ocid>",
     materialize=True
   )
   df = ds_reader.read()

Alternatively, you can use the ``.read_labeled_data()`` method by either specifying ``path`` or ``dataset_id``.

This example loads a labeled dataset and returns a Pandas dataframe containing the content and the annotations:

.. code:: ipython3

   df = pd.DataFrame.ads.read_labeled_data(
       path="<metadata_path>",
       materialize=True
   )

The following example loads a labeled dataset from the OCI Data Labeling, and returns a Pandas dataframe containing the content and the annotations:

.. code:: ipython3

   df = pd.DataFrame.ads.read_labeled_data(
       dataset_id="<dataset_ocid>",
       materialize=True
   )

