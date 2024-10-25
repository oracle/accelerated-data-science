============
Development
============

Data Formatting
---------------

Datetime Column
===============

Operators read data in "long" format, which requires a datetime column with a constant frequency (e.g., daily, quarterly, hourly). The operator will attempt to infer the datetime format, but if it's ambiguous, users can specify the format explicitly in the ``format`` field of ``datetime_column`` as shown below:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
            format: "%Y-%m-%d"
        historical_data:
            url: oci://<bucket_name>@<namespace_name>/example_yosemite_temps.csv
        horizon: 3
        target_column: y

Target Category Columns
=======================

A target category column, or series column, is optional. Use this field when you have multiple related forecasts over the same time period, such as predicting sales across different stores, forecasting system failures across multiple sensors, or forecasting different line items of a financial statement. The ``target_category_columns`` is a list of column names, though typically it contains just one. If a ``target_category_columns`` is specified in the ``historical_data``, it must also be present across all time periods in the ``additional_data``. Below is an example dataset and corresponding YAML:

Example Dataset:

=======  ========  ========
Product   Qtr      Sales
=======  ========  ========
A        01-2024   $7,500
B        01-2024   $4,500
C        01-2024   $8,500
A        04-2024   $9,500
B        04-2024   $6,500
C        04-2024   $9,500
=======  ========  ========

YAML Configuration:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: Qtr
            format: "%m-%Y"
        historical_data:
            url: historical_data.csv
        target_category_columns:
            - Product
        horizon: 1
        target_column: Sales

Additional Data
===============

Additional data enables multivariate forecasts and must adhere to similar formatting rules as historical data:

- It must include a datetime column with identical formatting to the historical data.
- If a target category column is present in the historical data, it must also be present in the additional data.
- The additional data must cover the entire forecast horizon.

Continuing with the previous example, for a horizon of 1, the additional data would look like this:

=======  ========  ========  ===================
Product   Qtr      Promotion  Competitor Release
=======  ========  ========  ===================
A        01-2024   0          0
B        01-2024   0          1
C        01-2024   1          1
A        04-2024   1          1
B        04-2024   0          0
C        04-2024   0          0
A        07-2024   0          0
B        07-2024   0          0
C        07-2024   0          0
=======  ========  ========  ===================

Corresponding YAML Configuration:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: Qtr
            format: "%m-%Y"
        historical_data:
            url: data.csv
        additional_data:
            url: additional_data.csv
        target_category_columns:
            - Product
        horizon: 1
        target_column: Sales

Output Directory
================

Before running operators on a job, users must configure their output directory. By default, results are output locally to a new folder named "results". This can be customized as shown below:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: oci://<bucket_name>@<namespace_name>/example_yosemite_temps.csv
        output_directory:
            url: oci://<bucket_name>@<namespace_name>/my_results/
        horizon: 3
        target_column: y

Ingesting and Interpreting Outputs
==================================

The forecasting operator generates several output files: ``forecast.csv``, ``metrics.csv``, ``local_explanations.csv``, ``global_explanations.csv``, and ``report.html``.

We will review each of these output files in turn.

**forecast.csv**

This file contains the entire historical dataset with the following columns:

- **Series**: Categorical or numerical index
- **Date**: Time series data
- **Real values**: Target values from historical data
- **Fitted values**: Model predictions on historical data
- **Forecasted values**: Predictions for the forecast horizon
- **Upper and lower bounds**: Confidence intervals for predictions (based on the specified confidence interval width in the YAML file)

**report.html**

The ``report.html`` file is customized for each model type. Generally, it contains a summary of the historical and additional data, plots of target values overlaid with fitted and forecasted values, analysis of the models used, and details about the model components. It also includes a "receipt" YAML file, providing a detailed version of the original ``forecast.yaml``.

**metrics.csv**

This file includes relevant metrics calculated on the training set.

**Global and Local Explanations in Forecasting Models**

Understanding the predictions and the driving factors behind them is crucial in forecasting models. Global and local explanations offer insights at different levels of granularity.

**Global Explanations:**

Global explanations provide a high-level overview of how a forecasting model operates across the entire dataset. Key aspects include:

1. **Feature Importance**: Identifies and ranks variables based on their contribution to the model's predictions.
2. **Model Structure**: Reveals the architecture, algorithms, parameters, and hyperparameters used in the model.
3. **Trends and Patterns**: Highlights broad trends and patterns captured by the model, such as seasonality and long-term trends.
4. **Assumptions and Constraints**: Uncovers underlying assumptions or constraints of the model.

**Local Explanations:**

Local explanations focus on specific data points or subsets, offering detailed insights into why the model made particular predictions. Key aspects include:

1. **Instance-specific Insights**: Provides details on how individual features contributed to a specific prediction.
2. **Contextual Understanding**: Considers the unique characteristics of the data point in question.
3. **Model Variability**: Shows the model's sensitivity to changes in input variables.
4. **Decision Boundaries**: In classification problems, explains the factors influencing specific classification outcomes.

Global explanations offer a broad understanding of the model, while local explanations provide detailed insights at the individual data point level.
