Quick Start
===========

This section provides a quick introduction to build a classifier using the Oracle AutoMLx tool for Iris dataset.
The dataset is a multi-class classification dataset, and more details about the dataset
can be found at `Iris dataset <https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_. We demonstrate
the preliminary steps required to train a model with the Oracle AutoMLx tool. We then explain the tuned model.

Load dataset
------------
We start by reading in the dataset from Scikit-learn.

.. code-block:: python3

    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> df = pd.DataFrame(data['data'], columns=data['feature_names'])
    >>> y = pd.Series(data['target'])

This toy dataset only contains numerical data. 
We now separate the predictions (`y`) from the training data (`X`) for both the training (`70%`) and test (`30%`) datasets.
The training set will be used to create a Machine Learning model using AutoMLx,
and the test set will be used to evaluate the model's performance on unseen data.

.. code-block:: python3

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(df,
                                                            y,
                                                            train_size=0.7,
                                                            random_state=0)
    >>> X_train.shape, X_test.shape
    ((105, 4), (45, 4))

Set the AutoMLx engine
----------------------
AutoMLx offers the `init() <http://automl.oraclecorp.com/multiversion/v23.1.1/initialization.html#automl.interface.init>`__ function, which allows to initialize the parallel engine.
By default, the AutoMLx pipeline uses the *dask* parallel engine. One can also set the engine to *local*,
which uses python's multiprocessing library for parallelism instead.

.. code-block:: python3

    >>> import automl
    >>> from automl import init
    
    >>> init(engine='local')
    [2023-01-12 05:48:31,814] [automl.xengine] Local ProcessPool execution (n_jobs=36)
 
Train a model using AutoMLx
---------------------------
The Oracle AutoMLx solution provides a pipeline that automatically finds a tuned model given a prediction task and a training dataset.
In particular it allows to find a tuned model for any supervised prediction task, e.g. classification or regression
where the target can be binary, categorical or real-valued.

AutoMLx consists of five main modules: 
    #. **Preprocessing** : Clean, impute, engineer, and normalize features.
    #. **Algorithm Selection** : Identify the right classification algorithm for a given dataset.
    #. **Adaptive Sampling** : Select a subset of the data samples for the model to be trained on.
    #. **Feature Selection** : Select a subset of the data features, based on the previously selected model.
    #. **Hyperparameter Tuning** : Find the right model parameters that maximize score for the given dataset. 

All these pieces are readily combined into a simple AutoMLx pipeline which
automates the entire Machine Learning process with minimal user input/interaction.

The AutoMLx API is quite simple to work with. We create a :obj:`automl.Pipeline<http://automl.oraclecorp.com/multiversion/v23.1.1/automl.html#automl.Pipeline>`__ instance.
Next, the training data is passed to the `fit() <http://automl.oraclecorp.com/multiversion/v23.1.1/automl.html#automl.Pipeline.fit>`__ function which executes the previously mentioned steps.

.. code-block:: python3

    >>> est = automl.Pipeline(task='classification')
    >>> est.fit(X_train, y_train)
        Pipeline()

A model is then generated (`est`) and can be used for prediction tasks. 
Here, we use the `F1_score` scoring metric to evaluate the performance of this model on unseen data (`X_test`).

.. code-block:: python3

    >>> from sklearn.metrics import f1_score
    >>> y_pred = est.predict(X_test)
    >>> score_default = f1_score(y_test, y_pred, average='macro')
    >>> print(f'Score on test data : {score_default}')
    Score on test data : 0.975983436853002


The `automl.Pipeline<http://automl.oraclecorp.com/multiversion/v23.1.1/automl.html#automl.Pipeline>`__ can also fit regression, forecasting and anomaly detection models.
Please check out the rest of the documentation for more details about advanced configuration parameters.

Explain a classifier
--------------------
For a variety of decision-making tasks, getting only a prediction as model output is not sufficient.
A user may wish to know why the model outputs that prediction, or which data features are relevant for that prediction. 
For that purpose the Oracle AutoMLx solution defines the `automl.interface.mlx.MLExplainer<http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#automl.interface.mlx.MLExplainer>`__ object, which allows to compute a variety of model explanations for any AutoMLx-trained pipeline or scikit-learn-like model.
`automl.interface.mlx.MLExplainer<http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#automl.interface.mlx.MLExplainer>`__ takes as argument the trained model, the training data and labels, as well as the task.

.. code-block:: python3

>>> explainer = automl.MLExplainer(est,
                                   X_train,
                                   y_train,
                                   task="classification")

Let's explain the model's performance (relative to the provided train labels) using Global Feature Importance. This technique would change
if a given feature were dropped from the dataset, without retraining the model.
This notion of feature importance considers each feature independently from all other features.

The method :obj:`explain_model() <automl.interface.mlx.MLExplainer.explain_model>` allows to compute such feature importances. It also provides 95% confidence intervals for each feature importance attribution.

    >>> result_explain_model_default = explainer.explain_model()
    >>> result_explain_model_default.to_dataframe()
    	feature	attribution	upper_bound	lower_bound
    0	petal width (cm)	0.350644	0.416850	0.284437
    1	petal length (cm)	0.272190	0.309005	0.235374
    2	sepal length (cm)	0.000000	0.000000	0.000000
    3	sepal width (cm)	0.000000	0.000000	0.000000

The oracle AutoMLx solution offers advanced configuration options and allows one to change the effect of feature interactions and interaction evaluations.
It also provides other model and prediction explanation techniques, such as:
  - `Local feature importance <http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#baselfiexplanation>`__, for example, using Kernel SHAP or an enhanced LIME;
  - `Feature Dependence Explanations <http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#fdexplanation>`__, such as partial dependence plots or accumulated local effects;
  - `Interactive What-IF explainers <http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#tabularexplainer>`__, which let users explore a model's predictions; and
  - `Counterfactual explanations <http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#cfexplanation>__`, which show how to change a row to obtain a desired outcome.
Please check out the `automl.interface.mlx.MLExplainer<http://automl.oraclecorp.com/multiversion/v23.1.1/mlx.html#automl.interface.mlx.MLExplainer>`__ documentation for more details.