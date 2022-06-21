Export Metadata
===============

To obtain a handle to a ``DataLabeling`` object, you call the ``DataLabeling()`` constructor. The default compartment is the same compartment as the notebook session, but the ``compartment_id`` parameter can be used to select a different compartment.

To work with the labeled data, you need a snapshot of the dataset. The ``export()`` method copies the labeled data from the Data Labeling service into a bucket in Object Storage. The ``.export()`` method has the following parameters:

- ``dataset_id``: The OCID of the Data Labeling dataset to take a snapshot of.
- ``path``: The Object Storage path to store the generated snapshot.

The export process creates a JSONL file that contains metadata about the labeled dataset in the specified bucket. 
There is also a record JSONL file that stores the image, text, or document file path of each record and its label.

The ``export()`` method returns the path to the metadata file that was created in the export operation.


.. code-block:: python3

   from ads.data_labeling import DataLabeling
   dls = DataLabeling()
   metadata_path = dls.export(
       dataset_id="<dataset_id>",
       path="oci://<bucket_name>@<namespace>/<prefix>"
   )


