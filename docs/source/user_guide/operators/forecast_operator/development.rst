============
Development
============

Data Formatting
---------------

Datetime Column
=================

Operators read data in the "long" format. There should be a datetime column with a constant frequency (i.e. daily, quarterly, hourly). The operator will attempt to guess the format, but if it's ambiguous, users can give the format explicitly under the ``format`` field of ``datetime_column`` as shown below:

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
========================

A target category column, or series column, is optional. Use this field when you have multiple related forecasts over the same time period. For example, predicting the sales across 10 different stores, or forecasting a system failure across 100 different sensors, or forecasting different line items of the same financial statement. The ``target_category_columns`` is a list of column names, although typically it's just 1. If a ``target_category_columns`` is specified in the ``historical_data``, it should also be available across all time periods in the ``additional_data``. See below for an example dataset and yaml:

=======  ========  ======== 
Product   Qtr       Sales
=======  ========  ======== 
A        01-2024    $7,500 
B        01-2024    $4,500  
C        01-2024    $8,500  
A        04-2024    $9,500 
B        04-2024    $6,500  
C        04-2024    $9,500  
=======  ========   ======== 

With the following yaml file:

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
================

Additional Data enables forecasts to be multivariate. Additional data follows similarly strict formatting to the historical data:

- It must have a datetime_column which has identical formatting to the historical data.
- If a target_category_column is present in the historical data, it must be present in the additional. 
- The additional data must contain data for the entire horizon. 

Following our example from above, for a horizon of 1, we would need the following additional_data:

=======  ========  ========  ===================
Product   Qtr      Promotion  Competitor Release
=======  ========  ========  ===================
A        01-2024    0          0
B        01-2024    0          1
C        01-2024    1          1
A        04-2024    1          1
B        04-2024    0          0
C        04-2024    0          0
A        07-2024    0          0
B        07-2024    0          0
C        07-2024    0          0
=======  ========   ======== ====================

And corresponding yaml file:

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

Before moving operators runs onto a job, users must configure their output directory. By default, results are output locally to a new folder "results". However, this can be specified directly as follows:

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


Ingesting and Interpretting Outputs
------------------------------------

The forecasting operator produces many output files: ``forecast.csv``, ``metrics.csv``, ``local_explanations.csv``, ``global_explanations.csv``, ``report.html``.

We will go through each of these output files in turn.

**Forecast.csv**

This file contains the entire historical dataset with the following columns:

- Series: Categorical or numerical index
- Date: Time series data
- Real values: Target values from historical data
- Fitted values: Model's predictions on historical data
- Forecasted values: Only available over the forecast horizon, representing the true forecasts
- Upper and lower bounds: Confidence intervals for the predictions (based on the specified confidence interval width in the YAML file)

**report.html**

The report.html file is designed differently for each model type. Generally, it contains a summary of the historical and additional data, a plot of the target from historical data overlaid with fitted and forecasted values, analysis of the models used, and details about the model components. It also includes a receipt YAML file, providing a fully detailed version of the original forecast.yaml file.

**Metrics.csv**

The metrics file includes relevant metrics calculated on the training set.


**Global and Local Explanations in Forecasting Models**

In the realm of forecasting models, understanding not only the predictions themselves but also the factors and features driving those predictions is of paramount importance. Global and local explanations are two distinct approaches to achieving this understanding, providing insights into the inner workings of forecasting models at different levels of granularity.

**Global Explanations:**

Global explanations aim to provide a high-level overview of how a forecasting model works across the entire dataset or a specific feature space. They offer insights into the model's general behavior, helping users grasp the overarching patterns and relationships it has learned. Here are key aspects of global explanations:

1. **Feature Importance:** Global explanations often involve the identification of feature importance, which ranks variables based on their contribution to the model's predictions. This helps users understand which features have the most significant influence on the forecasts.

2. **Model Structure:** Global explanations can also reveal the architecture and structure of the forecasting model, shedding light on the algorithms, parameters, and hyperparameters used. This information aids in understanding the model's overall approach to forecasting.

3. **Trends and Patterns:** By analyzing global explanations, users can identify broad trends and patterns in the data that the model has captured. This can include seasonality, long-term trends, and cyclical behavior.

4. **Assumptions and Constraints:** Global explanations may uncover any underlying assumptions or constraints the model operates under, highlighting potential limitations or biases.

While global explanations provide valuable insights into the model's behavior at a holistic level, they may not capture the nuances and variations that exist within the dataset.

**Local Explanations:**

Local explanations, on the other hand, delve deeper into the model's predictions for specific data points or subsets of the dataset. They offer insights into why the model made a particular prediction for a given instance. Key aspects of local explanations include:

1. **Instance-specific Insights:** Local explanations provide information about the individual features and their contribution to a specific prediction. This helps users understand why the model arrived at a particular forecast for a particular data point.

2. **Contextual Understanding:** They consider the context of the prediction, taking into account the unique characteristics of the data point in question. This is particularly valuable when dealing with outliers or anomalous data.

3. **Model Variability:** Local explanations may reveal the model's sensitivity to changes in input variables. Users can assess how small modifications to the data impact the predictions.

4. **Decision Boundaries:** In classification problems, local explanations can elucidate the decision boundaries and the factors that led to a specific classification outcome.

While local explanations offer granular insights, they may not provide a comprehensive understanding of the model's behavior across the entire dataset.
