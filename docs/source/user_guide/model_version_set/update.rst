Update
______

ModelVersionSet Properties
--------------------------

The ``ModelVersionSet`` object has a number of properties that you can be update. When the properties in a ``ModelVersionSet`` object are updated, the model version set in the model catalog are not automatically updated. You must call the ``.update()`` method to commit the changes.

The properties that you can be update are:

* ``compartment_id``: The OCID of the compartment that the model version set belongs to.
* ``description``: A description of the models in the collection.
* ``freeform_tags``: A dictionary of string values.
* ``name``: Name of the model version set.
* ``project_id``: The OCID of the data science project that the model version set belongs to.

The following demonstrates how to update these values of a model version set using the various API interfaces:

Parameter-based Pattern
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
    mvs.compartement_id = os.environ["PROJECT_COMPARTMENT_OCID"]
    mvs.description = "An updated description"
    mvs.freeform_tags = {'label_1': 'value 1', 'label_2': 'value 2'}
    mvs.name = "new_set_name"
    mvs.project_id = os.environ["PROJECT_OCID"]
    mvs.update()

Builder Pattern
^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
    mvs = (mvs.with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
              .with_description("An updated description")
              .with_freeform_tags(label_1="value 1", label_2="value 2")
              .with_name("new_set_name")
              .with_project_id(os.environ["PROJECT_OCID"])
              .update())


Version Label
-------------

The version label is associated with the model, and not the model version set. To change the version label, you must have a ``Model`` object. Then, you can change the ``version_label`` for the registered model.

The following example gets a registered ``Model`` object by model's OCID. Then, the object updates the version label property.

.. code-block:: python3

    from ads.model import LightGBMModel

    lgbm_model = LightGBMModel.from_id(
        "ocid1.datasciencemodel.oc1.xxx.xxxxx",
        model_file_name="model.joblib",
        artifact_dir="lgbm-download-test",
        force_overwrite=True,
    )

    lgbm_model.update(version_label="MyNewVersionLabel")