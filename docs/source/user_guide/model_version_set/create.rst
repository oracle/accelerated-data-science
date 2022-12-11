Create
______

The ``.create()`` method on a ``ModelVersionSet`` object creates a model version set in the model catalog. The properties of the ``ModelVersionSet`` are used to create the model version set in the model catalog. 

The following examples create a ``ModelVersionSet``, define the properties of the model version set, and then create a model version set in the model catalog.


Parameter-based Pattern
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = ModelVersionSet(
        compartment_id = os.environ["PROJECT_COMPARTMENT_OCID"],
        name = "my_model_version_set",
        projectId = os.environ["PROJECT_OCID"],
        description = "Sample model version set")
    mvs.create()


Builder Pattern
^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = (ModelVersionSet()
            .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
            .with_project_id(os.environ["PROJECT_OCID"])
            .with_name("my_model_version_set")
            .with_description("Sample model version set"))
    mvs.create()


