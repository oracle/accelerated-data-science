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

