=========================================================
Will the Anomaly Detection Operator Work for My Use Case?
=========================================================

As a low-code extensible framework, operators enable a wide range of use cases. This section will highlight some of the use cases that the AI Anomaly Detection Operator aims to serve.


**Dataset Size**

* First off, if you're unsure what model to use, we recommend using the "auto" setting, which is the default. "auto" will look at the parameters of your data and pick an algorithm that is likely to converge in a reasonable amount of time. Note, this may not always be the most performant algorithm! If need accuracy and do not care about cost or time, we recommend using all 5 frameworks and comparing across test datasets.
* When under 5,000 rows, and 5 columns, all operators should be quick, finishing in a couple minutes. If you include explainability, it may take longer.
* Over 5,000 rows, different algorithms perform to different degrees. This varies on more than the size of the dataset, but the service provides some recommendations in the next section, *Which Model is Right for You?*.
* For best results, the service recommends a minimum of 100 rows per category, however this is not a requirement, see "Cold Start Problem" below.
* For best results, the service recommends fewer than 100 total categories. Increasing category count is expected to linearly increase the time to completion.


**Which Model is Right for You?**

* The ARIMA and AutoMLX models slow down substantially as you increase columns. Aim to use these when you have less than 10 additional data columns.
* AutoTS is a global model. It works well for wide datasets but can take a long time to train especially on long datasets. One technique here is to pass ``model_list: superfast`` into the model kwargs to speed up an initial run.  To fully utilize autots, consider setting ``model_list: all`` in the ``model_kwargs``, however this may lead to the model taking a long time or even hanging.
* Prophet and NeuralProphet are much more consistent in their time to completion, and perform very well on most datasets.
* Automlx is not recommended when the data interval is less than 1 hour.
* Note: Explainability usually takes several minutes to a couple of hours. Explanations can be enabled using the flag ``generate_explanations: True``, however this is False by default. Because explanations are highly parallelized computations, explanations can be sped up by scaling up your compute shape.


**Datetime Input**

* The datetime input column must have a consistent interval throughout the historical and additional datasets. Inconsistent diffs will cause failure on automlx and may affect performance on other frameworks.
* It is strongly recommended that the datetime column is passed in sorted from earliest to latest, however this is not a requirement, and the operator will attempt to sort on your behalf.
* It is recommended that you pass in the format of your datetime string into the ``format`` option of the ``datetime_column`` parameter. The operator uses the python datetime string format outlined here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes


**Output Files**

* Apart from the ``report.html``, all output files should match formatting regardless of the model framework used (e.g. AutoMLX v Prophet).
* The ``report.html`` is custom built for each model framework, and so it will differ.


**Feature Engineering**

* The Operator will perform most  feature engineering on your behalf. 
