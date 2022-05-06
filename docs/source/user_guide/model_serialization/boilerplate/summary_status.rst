You can call the ``.summary_status()`` method after a model serialization instance such as ``AutoMLModel``, ``GenericModel``, ``SklearnModel``, ``TensorFlowModel``, or ``PyTorchModel`` is created. The ``.summary_status()`` method returns a Pandas dataframe that guides you through the entire workflow. It shows which methods are available to call and which ones aren't. Plus it outlines what each method does. If extra actions are required, it also shows those actions.

The following image displays an example summary status table created after a user initiates a model instance. The table's Step column displays a Status of Done for the initiate step. And the ``Details`` column explains what the initiate step did such as generating a ``score.py`` file. The Step column also displays  the ``prepare()``, ``verify()``, ``save()``, ``deploy()``, and ``predict()`` methods for the model. The Status column displays which method is available next. After the initiate step,  the ``prepare()`` method is available. The next step is to call the ``prepare()`` method. 

.. figure:: figure/summary_status.png
   :align: center
