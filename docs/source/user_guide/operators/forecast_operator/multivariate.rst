=====================
Multivariate Forecast
=====================

Structuring Data
----------------

Multivariate Forecasting is different to other multivariate machine learning problem types. When forecasting, all additional variables need be known over the full horizon. As such, the Forecast Operator **requires** additional_data for the entire horizon. (If you want to forecast the peak temperature for tomorrow, you can't use tomorrow's humidity - it's unknown!). For many enterprise problems this is not an issue: many retailers have long-term marketing projects with knowable future spends, Holidays are knowable in advance, etc. And sometimes our users will make assumptions for a what if analysis.

However, some variables are unknowable but still useful. For these we recommend *lagging*. To lag a variable, you will want to shift all of the values so that the horizon is full of data. (Typically users will shift by the entire horizon, but advanced users may have reasons to shift more than the horizon.) In essence, the operator is using the humidity from 5 days ago to predict the peak temperature today. 

Additional data should always have the same datetime column as the historical data plus the horizon. i.e. #rows(additional_data) = #rows(historical) + horizon.

If there is a target_category_column in the historical data, it should be present in the additional data as well.

As an example, if the historical data is:

====  ========= 
 Qtr   Revenue 
====  ========= 
 Q1    1200     
 Q2    1300  
 Q3    1500  
===  ======== 

Then the additional data (with a horizon=1) will need to be formatted as:

====  ========  ========  ==============
 Qtr    COGS    Discount   SP500 Futures
====  ========  ========  ==============
 Q1    100        0        1.02
 Q2    100        0.1      1.03
 Q3    105        0        1.04
 Q4    105        0.1      1.01
===  ========  ========  ==============


Note that the additional data does not contain the target column (Revenue), but it does contain the datetime column (Qtr). We would include this additional data in the yaml as shown below:

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

We can experiment with removing columns and examining how the results change. Below we ingest only 2 of the 3 additional columns.

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
