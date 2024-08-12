====================
AI Forecast Operator
====================

The Forecasting Operator leverages historical time series data to generate accurate forecasts for future trends. This operator aims to simplify and expedite the data science process by automating the selection of appropriate models and hyperparameters, as well as identifying relevant features for a given prediction task.

Power in Simplicity
===================

The Operator is simple to use, easy to extend, and as powerful as a team of data scieintists. To get started with the simplest forecast, use the following yaml file:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        target_column: y

We will extend this example in various ways throughout this documentation, however every parameter beyond those above are optional.


Modeling Options
================

There is no perfect model. A core feature of operators is model framework selection. For enterprise AI, usually 1 or 2 model frameworks perform best for your problem space. Each model framework is optimized for different assumptions of dataset size, frequency, complexity, seasonality, and more. The best way to determine which framework is right for you is to test on empirical data. Having worked across several enterprise forecasting problems, the ADS team has found the following frameworks to be the most effective. The model frameworks range from traditional statisical models, to complex machine learning models, all the way to deep neural networks:

- **Prophet**
- **ARIMA**
- **LightGBM**
- **NeuralProphet**
- **AutoTS**

Note: AutoTS is not a modeling framework, but a combination of many frameworks. AutoTS algorithms include (v 0.6.15): ConstantNaive, LastValueNaive, AverageValueNaive, GLS, GLM, ETS, ARIMA, FBProphet, RollingRegression, GluonTS, SeasonalNaive, UnobservedComponents, VECM, DynamicFactor, MotifSimulation, WindowRegression, VAR, DatepartRegression, UnivariateRegression, UnivariateMotif, MultivariateMotif, NVAR, MultivariateRegression, SectionalMotif, Theta, ARDL, NeuralProphet, DynamicFactorMQ, PytorchForecasting, ARCH, RRVAR, MAR, TMF, LATC, KalmanStateSpace, MetricMotif, Cassandra, SeasonalityMotif, MLEnsemble, PreprocessingRegression, FFT, BallTreeMultivariateMotif, TiDE, NeuralForecast, DMD

Auto-Select
-----------

For users that are completely new to forecasting, the operator also has an ``auto-select`` option. This will be the most computationally expensive option. The operator will split the training data into several validation sets, evaluate each framework, and attempt to determine the best. Auto-select does not guarentee to find the best, and is not recommended as a defualt configuration for end users due to its complexity.

Specify Model
-------------

Pick the desired model from the list above and insert it into the ``model`` parameter slot.


.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        model: <INSERT_MODEL_NAME_HERE>
        target_column: y


Evaluation and Explanation
==========================

As an Enterprise AI Solution, Operators know the evaluation and explanation of forecasts can be as critical as the forecasts themselves.

Reporting
---------

With every operator run, a report is generated to summarize the work done. The report includes the following:

- summary of the input data
- visualization of the forecast
- breakdown of major trends
- explanation (via shap values) of additional features
- table of metrics
- copy of the configuration yaml file.

Metrics
-------

Different use cases will optimizie for different metrics. The operator allows users to specify the metric they want to optimize over from the following list:

- MAPE
- RMSE
- SMAPE
- MSE

The metric is optinoally specified in the yaml file:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        model: prophet
        target_column: y
        metric: rmse


Explanations
------------

When additional data is provided, the operator will optionally provide explanations for these additional features (columns). Explanations are generated from shap values. Users can enable explanations from the yaml file:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        model: prophet
        target_column: y
        generate_explanations: True


With large datasets shap values can be expensive to generate. Enterprise applications may vary in their need for decimal accuracy versus cost. Therefore the operator offers several options:

- FAST_APPROXIMATE: (default) Generated shap values are typically within 1% of the true values and takes 1% of the time.
- BALANCED: Generated shap values are typically within 0.1% of the true values and takes 10% of the time.
- HIGH_ACCURACY: Generates the true shap values at full precision.

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        model: prophet
        target_column: y
        generate_explanations: True
        explanations_accuracy_mode: BALANCED

Selecting the best accuracy mode will require empircal testing, but ``FAST_APPROXIMATE`` is usually enough for real-world data.

Note: The above example won't generate explanations because there is no additional data. The shap values would be 100% for the feature ``y``.


.. toctree::
  :maxdepth: 1

  ./quickstart
  ./use_cases
  ./install
  ./productionize
  ./advanced_use_cases
  ./yaml_schema
  ./faq
