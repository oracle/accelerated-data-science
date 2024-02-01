==================
Advanced Use Cases
==================

**Documentation: Anomaly Detection Science and Model Parameterization**

**The Science of Anomaly Detection**

Forecasting is a complex yet essential discipline that involves predicting future values or events based on historical data and various mathematical and statistical techniques. To achieve accurate forecasts, it is crucial to understand some fundamental concepts:

**Seasonality**

Seasonality refers to patterns in data that repeat at regular intervals, typically within a year. For example, retail sales often exhibit seasonality with spikes during holidays or specific seasons. Seasonal components can be daily, weekly, monthly, or yearly, and understanding them is vital for capturing and predicting such patterns accurately.

**Stationarity**

Stationarity is a critical property of time series data. A time series is considered stationary when its statistical properties, such as mean, variance, and autocorrelation, remain constant over time. Stationary data simplifies forecasting since it allows models to assume that future patterns will resemble past patterns.

**Cold Start**

The "cold start" problem arises when you have limited historical data for a new product, service, or entity. Traditional forecasting models may struggle to make accurate predictions in these cases due to insufficient historical context.

**Passing Parameters to Models**

To enhance the accuracy and adaptability of forecasting models, our system allows you to pass parameters directly. Here's how to do it:


**Specify Model Type**

Sometimes users will know which models they want to use. When users know this in advance, they can specify using the ``model_kwargs`` dictionary. In the following example, we will instruct the model to *only* use the ``DecisionTreeRegressor`` model.

.. code-block:: yaml

  kind: operator
  type: forecast
  version: v1
  spec:
    model: automlx
    model_kwargs:
      model_list:
        - NaiveForecaster
      search_space:
        NaiveForecaster:
          sp: [1,100]


When using autots, there are model_list *families*. These families are named after the shared characteristics of the models included. For example, we can use the autots "superfast" model_list and set it in the following way:

.. code-block:: yaml

  kind: operator
  type: forecast
  version: v1
  spec:
    model: autots
    model_kwargs:
      model_list: superfast


Note: this is only supported for the ``autots`` model.


**Specify Other Model Details**

In addition to ``model_list``, there are many other parameters that can be specified. Users may specify, for example, the search space they want to search for their given model type. In automlx, specifying a hyperparameter range is as simple as:

.. code-block:: yaml

  kind: operator
  type: forecast
  version: v1
  spec:
    model: automlx
    model_kwargs:
      search_space:
        LogisticRegression:
          C: 
            range: [0.03125, 512]
            type': continuous
          solver:
            range: ['newton-cg', 'lbfgs', 'liblinear', 'sag']
            type': categorical
          class_weight:
            range: [None, 'balanced']
            type: categorical


**When Models Perform Poorly and the "Auto" Method**

Forecasting models are not one-size-fits-all, and some models may perform poorly under certain conditions. Common scenarios where models might struggle include:

- **Sparse Data:** When there's limited historical data available, traditional models may have difficulty making accurate predictions, especially for cold start problems.

- **High Seasonality:** Extremely seasonal data with complex patterns can challenge traditional models, as they might not capture all nuances.

- **Non-Linear Relationships:** In cases where the relationships between input variables and forecasts are nonlinear, linear models may underperform.

- **Changing Dynamics:** If the underlying data-generating process changes over time, static models may fail to adapt.

Our system offers an "auto" method that strives to anticipate and address these challenges. It dynamically selects the most suitable forecasting model and parameterizes it based on the characteristics of your data. It can automatically detect seasonality, stationarity, and cold start issues, then choose the best-fitting model and adjust its parameters accordingly.

By using the "auto" method, you can rely on the system's intelligence to adapt to your data's unique characteristics and make more accurate forecasts, even in challenging scenarios. This approach simplifies the forecasting process and often leads to better results than manual model selection and parameter tuning.
