====================
AI Forecast Operator
====================

The AI Forecast Operator leverages historical time series data to generate accurate forecasts for future trends. This operator simplifies and accelerates the data science process by automating model selection, hyperparameter tuning, and feature identification for a given prediction task.

Power in Simplicity
===================

The Operator is designed to be simple to use, easy to extend, and as powerful as a team of data scientists. To get started with the simplest forecast, use the following YAML configuration:

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

We will extend this example in various ways throughout this documentation. However, all parameters beyond those shown above are optional.

Modeling Options
================

There is no perfect model. A core feature of the Operator is the ability to select from various model frameworks. For enterprise AI, typically one or two frameworks perform best for your problem space. Each model is optimized for different assumptions, such as dataset size, frequency, complexity, and seasonality. The best way to determine which framework is right for you is through empirical testing. Based on experience with several enterprise forecasting problems, the ADS team has found the following frameworks to be the most effective, ranging from traditional statistical models to complex machine learning and deep neural networks:

- **Prophet**
- **ARIMA**
- **AutoMLx**
- **MLForecast**
- **NeuralProphet**
- **AutoTS**

*Note: AutoTS is not a single modeling framework but a combination of many. AutoTS algorithms include (v0.6.15): ConstantNaive, LastValueNaive, AverageValueNaive, GLS, GLM, ETS, ARIMA, FBProphet, RollingRegression, GluonTS, SeasonalNaive, UnobservedComponents, VECM, DynamicFactor, MotifSimulation, WindowRegression, VAR, DatepartRegression, UnivariateRegression, UnivariateMotif, MultivariateMotif, NVAR, MultivariateRegression, SectionalMotif, Theta, ARDL, NeuralProphet, DynamicFactorMQ, PytorchForecasting, ARCH, RRVAR, MAR, TMF, LATC, KalmanStateSpace, MetricMotif, Cassandra, SeasonalityMotif, MLEnsemble, PreprocessingRegression, FFT, BallTreeMultivariateMotif, TiDE, NeuralForecast, DMD.*

Auto-Select
-----------

For users new to forecasting, the Operator also has an ``auto-select`` option. This is the most computationally expensive option as it splits the training data into several validation sets, evaluates each framework, and attempts to determine the best one. However, auto-select does not guarantee to find the optimal model and is not recommended as the default configuration for end-users due to its complexity.

Specify Model
-------------

You can manually select the desired model from the list above and insert it into the ``model`` parameter slot.

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

As an enterprise AI solution, the Operator ensures that the evaluation and explanation of forecasts are as critical as the forecasts themselves.

Reporting
---------

With every operator run, a report is generated to summarize the work done. The report includes:

- Summary of the input data
- Visualization of the forecast
- Breakdown of major trends
- Explanation (via SHAP values) of additional features
- Table of metrics
- A copy of the configuration YAML file

Metrics
-------

Different use cases optimize for different metrics. The Operator allows users to specify the metric they want to optimize from the following list:

- MAPE
- RMSE
- SMAPE
- MSE

The metric can be optionally specified in the YAML file:

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

When additional data is provided, the Operator can optionally generate explanations for these additional features (columns) using SHAP values. Users can enable explanations in the YAML file:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_pedestrians_covid.csv
        additional_data:
            url: additional_data.csv
        horizon: 3
        model: prophet
        target_column: y
        generate_explanations: True

With large datasets, SHAP values can be expensive to generate. Enterprise applications may vary in their need for decimal accuracy versus computational cost. Therefore, the Operator offers several options:

- **FAST_APPROXIMATE** (default): Generated SHAP values are typically within 1% of the true values and require 1% of the time.
- **BALANCED**: Generated SHAP values are typically within 0.1% of the true values and require 10% of the time.
- **HIGH_ACCURACY**: Generates the true SHAP values at full precision.

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

Selecting the best accuracy mode will require empirical testing, but ``FAST_APPROXIMATE`` is usually sufficient for real-world data.

*Note: The above example won't generate explanations because there is no additional data. The SHAP values would be 100% for the feature ``y``.*

.. toctree::
  :maxdepth: 1

  ./quickstart
  ./data_sources
  ./scalability
  ./multivariate
  ./install
  ./development
  ./use_cases
  ./yaml_schema
  ./faq
