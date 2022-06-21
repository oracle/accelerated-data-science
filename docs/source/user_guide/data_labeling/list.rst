List
****

The ``.list_dataset()`` method generates a list of the available labeled datasets in the compartment. The compartment is set when you call ``DataLabeling()``. The ``.list_dataset()`` method returns a Pandas dataframe where each row is a dataset.

.. code-block:: python3

   from ads.data_labeling import DataLabeling
   dls = DataLabeling(compartment_id="<compartment_id>")
   dls.list_dataset()


