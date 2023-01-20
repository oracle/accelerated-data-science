Model Version Set
*****************

Overview
________

The normal workflow of a data scientist is to create a model and push it into production. While in production the data scientist learns what the model is doing well and what it isn't. Using this information the data scientist creates an improved model. These models are linked using model version sets. A model version set is a collection of models that are related to each other. A model version set is a way to track the relationships between models. As a container, the model version set takes a collection of models. Those models are assigned a sequential version number based on the order they are entered into the model version set. 

In ADS the class ``ModelVersionSet`` is used to represent the model version set. An object of ``ModelVersionSet`` references a model version set in the Data Science service. The ``ModelVersionSet`` class supports two APIs: the builder pattern and the traditional parameter-based pattern. You can use either of these API frameworks interchangeably and examples for both patterns are included.

Use the ``.create()`` method to create a model version set in your tenancy. If the model version set already exists in the model catalog, use the ``.from_id()`` or ``.from_name()`` method to get a ``ModelVersionSet`` object based on the specified model version set. If you make changes to the metadata associated with the model version set, use the ``.update()`` method to push those changes to the model catalog. The ``.list()`` method lists all model version sets. To add an existing model to a model version set, use the ``.add_model()`` method. The ``.models()`` method lists the models in the model version set. Use the ``.delete()`` method to delete a model version set from the model catalog.


Quick Start
___________

The following creates a model and model version set, and then performs some common operations on the model version set:

.. code-block:: python3

    import tempfile
    from ads.model import SklearnModel
    from ads.model import ModelVersionSet
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Create a model version set
    mvs = ModelVersionSet(
        name = "my_test_model_version_set",
        description = "A test creating the model version set using ModelVersionSet")
    mvs.create()

    # Create a Sklearn model
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)


    # Create an SklearnModel object
    sklearn_model = SklearnModel(estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp())
    sklearn_model.prepare(inference_conda_env="dbexp_p38_cpu_v1")

    # Save the model and add it to the model version set
    model_id = sklearn_model.save(
        display_name="Quickstart model",
        model_version_set=mvs,
        version_label="Version 1")

    # Print a list of models in the model version set
    for item in ModelVersionSet.list():
        print(item)
        print("---------")

    # Update the model version set
    mvs.description = "Updated description of the model version set"
    mvs.update()

    # Delete the model version set and associated models
    # mvs.delete(delete_model=True)

Associate a Model
_________________

Model version sets are a collection of models. After a model is associated with a model version set, the model can't be associated with a different model version set. Further, the model can't be disassociated with the model version set.

When a model is associated with a model version set, a version label can be assigned to the set. This version is different than the model version that is maintained by the model version set.

There are a number of ways to associate a model with a model version set. Which approach you use depends on the workflow.

ModelVersionSet Object
----------------------

For a model not associated with a model version set,  use the ``.model_add()`` method on a ``ModelVersionSet`` object to associate the model with the model version set. The ``.model_add()`` requires that you provide the model OCID and optionally a version label.

.. code-block:: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
   mvs.model_add(<your_model_id>, version_label="Version 1")

Model Serialization
-------------------

The :ref:`Model Serialization` classes allow a model to be associated with a model version set at the time that it is saved to the model catalog. You do this with the ``model_version_set`` parameter in the ``.save()`` method. In addition, you can add the model's version label with the ``version_label`` parameter.

The ``model_version_set`` parameter accepts a model version set's OCID or name. The parameter also accepts a ``ModelVersionSet`` object.

In the following, the ``model`` variable is a :ref:`Model Serialization` object that is to be saved to the model catalog, and at the same time associated with a model version set.

.. code-block:: python3

   model.save(
       display_name='Model attached to a model version set',
       version_label = "Version 1",
       model_version_set="<model_version_set_id>"
   )


Context Manager
---------------

To associate several models with a model version set, use a context manager. The ``ads.model.experiment()`` method requires a ``name`` parameter. If the model catalog has a matching model version set name, the model catalog uses that model version set. If the parameter ``create_if_not_exists`` is ``True``, the ``experiment()`` method attempts to match the model version set name with name in the model catalog. If the name does not exist, the method creates a new model version set.

Within the context manager, you can save multiple :ref:`Model Serialization` models without specifying the ``model_version_set`` parameter because it's taken from the model context manager. The following example assumes that ``model_1``, ``model_2``, and ``model_3`` are :ref:`Model Serialization` objects. If the model version set doesn't exist in the model catalog, the example creates a model version set named ``my_model_version_set``.  If the model version set exists in the model catalog, the models are saved to that model version set.

.. code-block:: python3

   with ads.model.experiment(name="my_model_version_set", create_if_not_exists=True):
        # experiment 1
        model_1.save(
            display_name='Generic Model Experiment 1',
            version_label = "Experiment 1"
        )

        # experiment 2
        model_2.save(
            display_name='Generic Model Experiment 2',
            version_label = "Experiment 2"
        )

        # experiment 3
        model_3.save(
            display_name='Generic Model Experiment 3',
            version_label = "Experiment 3"
        )

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


Delete
______

To delete a model version set, all the associated models must be deleted or in a terminated state. You can set the ``delete_model`` parameter to ``True`` to delete all of the models in the model version set, and then delete the model version set. The ``.delete()`` method on a ``ModelVersionSet`` object initiates an asynchronous delete operation. You can check the ``.status`` method on the ``ModelVersionSet`` object to determine the status of the delete request. 


The following example deletes a model version set and its associated models. 

.. code-block: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
   mvs.delete(delete_model=True)


The ``status`` property has the following values:

* ``ModelVersionSet.LIFECYCLE_STATE_ACTIVE``
* ``ModelVersionSet.LIFECYCLE_STATE_DELETED``
* ``ModelVersionSet.LIFECYCLE_STATE_DELETING``
* ``ModelVersionSet.LIFECYCLE_STATE_FAILED``

Download
________

Create a ``ModelVersionSet`` object by downloading the metadata from the model catalog. The ``ModelVersionSet`` class has a ``.from_id()`` method that accepts the model version set OCID. The ``.from_name()`` method takes the name of the model version set.

``.from_id()``
^^^^^^^^^^^^^^

.. code-block:: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")


``.from_name()``
^^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = ModelVersionSet.from_name(name="<model_version_set_name>")

List
____

ModelVersionSet
---------------

The ``.list()`` method on the ``ModelVersionSet`` class takes a compartment ID and lists the model version sets in that compartment. If the compartment isn't given, then the compartment of the notebook session is used.

The following example uses context manager to iterate over the collection of model version sets:

.. code-block:: python3

    for model_version_set in ModelVersionSet.list():
        print(model_version_set)
        print("---------")


Model
-----

You can get the list of models associated with a model version set by calling the ``.models()`` method on a ``ModelVersionSet`` object. A list of models that are associated with that model version set is returned. First, you must obtain a ``ModelVersionSet`` object. Use the ``.from_id()`` method if you know the model version set OCID. Alternatively, use the ``.from_name()`` method if you know the name of the model version set.

.. code-block:: python3

    mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
    models = mvs.models()

    for dsc_model in models:
        print(dsc_model.display_name, dsc_model.id, dsc_model.status)


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


