=================================================
Will the Forecast Operator Work for My Use Case?
=================================================

As a low-code, extensible framework, the AI Forecast Operator supports a wide range of use cases. This section highlights key considerations for determining if the Forecast Operator is suitable for your needs.

Dataset Size
------------

- If you're unsure which model to use, we recommend starting with the default "auto" setting. This setting selects an algorithm based on your data's parameters, aiming for a reasonable convergence time. Note that "auto" may not always choose the most accurate algorithm. If accuracy is critical, and cost or time is not a concern, consider running all five frameworks and comparing results across test datasets.
- For datasets with fewer than 5,000 rows and 5 columns, all operators should complete quickly, typically within a few minutes. However, including explainability features may increase runtime.
- For datasets with more than 5,000 rows, performance varies by algorithm. Recommendations for selecting the appropriate model based on dataset size are provided in the next section, *Which Model is Right for You?*.
- For optimal results, we recommend a minimum of 100 rows per category, although this is not a strict requirement (see "Cold Start Problem" below).
- The service recommends fewer than 100 total categories for best performance. Increasing the number of categories is expected to linearly increase the time to completion.

Which Model is Right for You?
-----------------------------

- **ARIMA and AutoMLX**: These models slow down significantly as the number of columns increases. They are best suited for datasets with fewer than 10 additional data columns.
- **AutoTS**: A global model that works well with wide datasets but can take a long time to train, especially on long datasets. For faster initial runs, consider passing ``model_list: superfast`` in the model kwargs. To fully utilize AutoTS, set ``model_list: all`` in the ``model_kwargs``; however, this may significantly increase runtime or cause the model to hang.
- **Prophet and NeuralProphet**: These models are more consistent in their completion times and perform well on most datasets.
- **AutoMLX**: Not recommended for datasets with intervals shorter than 1 hour.
- **Explainability**: Generating explanations can take several minutes to hours. Explanations are disabled by default (``generate_explanations: False``). Enabling them (``generate_explanations: True``) and scaling up your compute shape can speed up this highly parallelized computation.

Target Column
-------------

- The target column must be present in the dataset specified in the ``historical_data`` field.
- The ``historical_data`` dataset must include: 1) a target column, 2) a datetime column, and optionally, 3) a target_category_column or series.
- The ``historical_data`` dataset should not contain any other columns.
- If you include ``additional_data``, it must have the same datetime column, the target_category_column (if present in the historical data), and any other required additional features.
- The ``additional_data`` dataset should not contain the target column.

Additional Features
-------------------

- Including additional "future regressors" is recommended when available, as these features can significantly improve forecasting accuracy.
- A "future regressor" is a variable known for all future timestamps within your forecast horizon at the time of training (e.g., decisions about product discounts or staffing levels).
- All additional data must be stored separately and passed into the "additional_data" field in the ``forecast.yaml`` file.
- Ensure that all additional data covers each period in the forecast horizon. Missing values may result in suboptimal forecasts.

Long Horizon Problems
---------------------

- A Long Horizon Problem occurs when the forecast horizon exceeds 50% of the historical data period (e.g., forecasting the next 2 years based on 4 years of data). These problems are particularly challenging for AutoMLX, AutoTS, and ARIMA. We recommend using NeuralProphet and/or Prophet for such scenarios.

Cold Start Problems
-------------------

- A cold start problem arises when there is data for some categories of the target variable but not all. The model can use proxies to forecast categories it has not encountered, based on trends and characteristics of the additional data.
- For cold start problems, we strongly recommend using AutoTS, as it is a global model that can ensemble multiple models into a single aggregate model, leveraging all dataset features to make predictions.

Datetime Input
--------------

- The datetime column must have a consistent interval throughout both the historical and additional datasets. Inconsistent intervals will cause failures in AutoMLX and may affect performance in other frameworks.
- Missing data is acceptable but may lead to suboptimal forecasts.
- It is strongly recommended that the datetime column be sorted from earliest to latest, although the operator will attempt to sort it if not.
- We recommend specifying the format of your datetime string using the ``format`` option in the ``datetime_column`` parameter. The operator follows the Python datetime string format guidelines found here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes.

Output Files
------------

- Apart from ``report.html``, all output files should follow a consistent format, regardless of the model framework used (e.g., AutoMLX vs. Prophet).
- The ``report.html`` file is custom-built for each model framework and will differ accordingly.
- All output files can be disabled except for ``forecast.csv``. To learn more about disabling output files, refer to the ``generate_X`` boolean parameters in the ``forecast.yaml`` file.

Feature Engineering
-------------------

- Except for ARIMA, avoid creating features based on "day of the week" or "holiday" as NeuralProphet, Prophet, AutoTS, and AutoMLX can generate this information internally.
- AutoMLX performs extensive feature engineering on your behalf, expanding features into lag, min, max, average, and more. When using AutoMLX, it is recommended to only pass features that contain new information.
- AutoTS also performs some feature engineering, though it is less extensive than AutoMLX.

The Science of Forecasting
--------------------------

Forecasting is a complex yet essential discipline that involves predicting future values or events based on historical data and various mathematical and statistical techniques. Understanding the following concepts is crucial for accurate forecasting:

**Seasonality**

- Seasonality refers to patterns in data that repeat at regular intervals, typically within a year (e.g., retail sales spikes during holidays). Understanding and accurately capturing these patterns is essential for effective forecasting.

**Stationarity**

- Stationarity is a critical property of time series data, where statistical properties like mean, variance, and autocorrelation remain constant over time. Stationary data simplifies forecasting by allowing models to assume that future patterns will resemble past patterns.

**Cold Start**

- The "cold start" problem occurs when there is limited historical data for a new product, service, or entity. Traditional forecasting models may struggle in these cases due to insufficient historical context.

**Passing Parameters to Models**

- Our system allows you to pass parameters directly to enhance the accuracy and adaptability of forecasting models.

Data Parameterization
---------------------

**Read Data from the Database**

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        historical_data:
            connect_args:
                user: XXX
                password: YYY
                dsn: "localhost/orclpdb"
            sql: 'SELECT Store_ID, Sales, Date FROM live_data'
        datetime_column:
            name: ds
        horizon: 1
        target_column: y

**Read Part of a Dataset**

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        historical_data:
            url: oci://bucket@namespace/data
            format: tsv
            limit: 1000  # Only the first 1000 rows
            columns: ["y", "ds"]  # Ignore other columns
        datetime_column:
            name: ds
        horizon: 1
        target_column: y

Model Parameterization
----------------------

When using AutoTS, there are *model_list* families, which group models based on shared characteristics. For example, to use the "superfast" model_list in AutoTS, configure it as follows:

.. code-block:: yaml

  kind: operator
  type: forecast
  version: v1
  spec:
    model: autots
    model_kwargs:
      model_list: superfast

Note: This configuration is supported only for the ``autots`` model.
