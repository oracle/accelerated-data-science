=================================================
Will the Forecast Operator Work For My Use Case?
=================================================

As a low-code extensible framework, operators enable a wide range of use cases. This section will highlight some of the use cases that the AI Forecast Operator aims to serve.


**Dataset Size**

* First off, if you're unsure what model to use, we recommned using the "auto" setting, which is the default. "auto" will look at the parameters of your data and pick an algorithm that is likely to converge in a reasonable amount of time. Note, this may not always be the most performant algorithm! If you are in need of accuracy and do not care about cost or time, we recommend using all 5 frameworks and comparing across test datasets.
* When under 5,000 rows, and 5 columns, all operators should be realtively quick, finishing in a couple minutes. If you include explainability, it may take longer.
* Over 5,000 rows, different algorithms perform to different degrees. This varies on more than simplay the size of the dataset, but the service provides some recomendations in the next section, __Which Model is Right for You?__.
* For best results, the service recommends a minimum of 100 rows per category, however this is not a requirement, see "Cold Start Problem" below.
* For best results, the service recommends fewer than 100 total categories. Increasing category count is expected to linearly increase the time to completion.


**Which Model is Right for You?**

* the ARIMA and AutoMLX models slow down substantially as you increase columns. Generally aim to use these when you have less than 10 additional data columns
* AutoTS is a global model. It works well for wide datasets, but can take a long time to train especially on long datasets. One technique here is to pass ``model_list: superfast`` into the model kwargs to speed up an inital run.  To fully utilize autots, consider setting ``model_list: all`` in the ``model_kwargs``, however this may lead to the model taking a long time or even hanging.
* Prophet and NeuralProphet are much more consistent in their time to completion, and perform very well on most datasets.
* Automlx is not recommended when the data interval is less than
* Note: Explainability usually takes several minutes to a couple of hours. Explanations can be enabled using the flag ``generate_explanations: True``, however this is False by default. Because Explanations are highly parallelized computations, speeding up the explanations generation is as simple as scaling up your compute shape.


**Target Column**

* The target column should be present in the dataset passed into the ``historical_data`` field. 
* The ``historical_data`` dataset must have 1. a target column, 2. a datetime column, and optionally 3. a target_category_column or series.
* The ``historical_data`` cannot contain any other columns.
* If passing ``additional_data``, it must match have the datetime column, the target_category_column if it's present in the historical data, and then as many additional features as needed.
* The ``additional_data`` cannot contain the target column.


**Additonal Features**

* It is recommended to include addtional "future regressors" when available. These features can greatly improve the ability to forecast.
* A "future regressor" is one that is known for all future timestamps in your forecast horizon during training time. (Typically these are variables within your control, such as whether or not to discount a product or the staffing of a particular outlet.)
* All additional data provided must be put in a separte location and passed into "additional_data" in the ``forecast.yaml`` file.
* All additional data must be given for each period of the forecast horizon. Missing values may result in sub-optimal forecasts.


**Long Horizon Problems**

* A Long Horizon Problem is defined by having a forecast horizon period that's more than 50% of the length of the input data period (e.g. forecasting next 2 years on 4 years of data). These problems are particularly difficult for AutoMLX, AutoTS, and ARMIA. Customers are encouraged to use NeuralProphet and/or Prophet for these types of problems. 


**Cold Start Problems**

* A cold start problem can occur when there's data available for some categories of the target variable, but not all. Using these proxies, the model can make a forecast for the categories it hasn't seen yet based on the trands and the additional data characteristcs. 
* For cold start problems, customers are strongly encouraged to use AutoTS as AutoTS is a "global model" implementation. AutoTS can ensemble many models into a single aggregate model, allowing it to rely on all features of the dataset in making any 1 prediction.


**Datetime Input**

* The datetime input column must have a consistent interval throughout the historical and additional datasets. Inconsistent diffs will cause failure on automlx and may affect performance on other frameworks.
* Note: missing data is okay, however it will likely cause sub-optimal forecasting.
* It is strongly recommended that the datetime column is passed in sorted from earliest to latest, however this is not a requirement, and the operator will atempt to sort on your behalf.
* It is recommended that you pass in the format of your datetime string into the ``format`` option of the ``datetime_column`` parameter. The operator uses the python datetime string format outlined here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes


**Output Files**

* With the exception of the ``report.html``, all output files should match formatting regardless of the model framework used (e.g. AutoMLX v Prophet).
* The ``report.html`` is custom built for each model framework, and so it will differ.
* All output files can be disabled, with the exception of ``forecast.csv``. For more details in disabling, look for ``generate_X`` boolean parameters in the ``forecast.yaml`` file.


**Feature Engineering**

* With the exception of ARIMA, it is not recommended to create features around "day of the week" or "holiday" as NeuralProphet, Prophet, AutoTS and AutoMLX can generate this ionformation internally.
* AutoMLX performs further feature engineering on your behalf. It will expand your features into lag, min, max, average, and more. When using automlx, it is recommended that you only pass in features that contain new information.
* AutoTS performs some feature engineering, but not as extensive as AutoMLX.
