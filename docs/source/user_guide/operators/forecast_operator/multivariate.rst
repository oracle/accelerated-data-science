=====================
Multivariate Forecast
=====================

Structuring Data
----------------

Multivariate forecasting differs from other multivariate machine learning problems. In forecasting, all additional variables must be known over the entire forecast horizon. Consequently, the Forecast Operator **requires** ``additional_data`` to cover the full horizon. For example, if you're forecasting the peak temperature for tomorrow, you cannot use tomorrow's humidity because it's unknown. However, many enterprise scenarios do not face this issue, as retailers often have long-term marketing plans with knowable future expenditures, holidays are predictable, etc. In some cases, users might make assumptions for a "what-if" analysis.

Sometimes, variables are useful but unknowable in advance. For these cases, we recommend *lagging* the variable. To lag a variable, shift all its values so that the horizon is filled with data. Typically, users shift by the entire horizon, though advanced users may shift by more or less depending on their needs. Essentially, the operator uses the humidity from five days ago to predict today's peak temperature.

The additional data must always share the same datetime column as the historical data and must extend beyond the horizon. In other words, the number of rows in ``additional_data`` should equal the number of rows in the historical data plus the horizon.

If the historical data includes a ``target_category_column``, it should also be present in the additional data.

For example, if the historical data is:

====  ========= 
 Qtr   Revenue 
====  ========= 
 Q1    1200     
 Q2    1300  
 Q3    1500  
====  ========= 

Then the additional data (with a horizon of 1) should be formatted as:

====  ========  ========  ==============
 Qtr    COGS    Discount   SP500 Futures
====  ========  ========  ==============
 Q1    100        0        1.02
 Q2    100        0.1      1.03
 Q3    105        0        1.04
 Q4    105        0.1      1.01
====  ========  ========  ==============

Note that the additional data does not include the target column (Revenue), but it does include the datetime column (Qtr). You would include this additional data in the YAML file as follows:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: Qtr
        historical_data:
            url: historical_data.csv
        additional_data:
            url: additional_data.csv
        horizon: 1
        model: prophet
        target_column: Revenue

You can experiment by removing columns and observing how the results change. Below is an example of ingesting only two of the three additional columns:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: Qtr
        historical_data:
            url: historical_data.csv
        additional_data:
            url: additional_data.csv
            columns:
                - Discount
                - SP500 Futures
        horizon: 1
        model: prophet
        target_column: Revenue
