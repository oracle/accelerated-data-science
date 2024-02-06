=========================================================
Will the Anomaly Detection Operator Work for My Use Case?
=========================================================

As a low-code extensible framework, operators enable a wide range of use cases. This section will highlight some of the use cases that the AI Anomaly Detection Operator aims to serve.


**Dataset Size**

* Datasets can be very large, upwards of 10M rows without issue on a relatively small shape.
* Increasing compute shape and memory will improve latency in the Operator.


**Which Model is Right for You?**

* Autots is a very comprehensive framework for time series data, winning the M6 benchmark. Parameters can be sent directly to AutoTS' AnomalyDetector class through the ``model_kwargs`` section of the yaml file.
* AutoMLX is a propreitary modeling framework developed by Oracle's Labs team and distributed through OCI Data Science. Parameters can be sent directly to AutoMLX's AnomalyDetector class through the ``model_kwargs`` section of the yaml file.
* Together these 2 frameworks train and tune more than 25 models, and deliver the est results.


**Datetime Input**

* The datetime input column must have a consistent interval throughout the historical and additional datasets. Inconsistent diffs will cause failure on automlx and may affect performance on other frameworks.
* It is recommended that the datetime column is passed in sorted from earliest to latest, if not, the operator will sort on your behalf.
* It is recommended that you pass in the format of your datetime string into the ``format`` option of the ``datetime_column`` parameter. The operator uses the python datetime string format outlined here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes (For example "31/02/2024" would be "%m/%d/%Y". "July 24" would be "%B %y".)


**Validation Data**

* Validation Data is optional, but when provided can improve the accuracy of the anomaly detection.
* Validation Data should have the all the same columns of the input_data *plus* a column titled ``anomaly`` that is either 1 (not an anomaly) or 0 (anomaly)


**Output Files**

* Apart from the ``report.html``, all output files will match formatting regardless of the model framework used.
* The ``report.html`` is custom built for each model framework, and so it will differ.


**Feature Engineering**

* The Operator will perform most feature engineering on your behalf, such as infering holidays, day of week, 


**Latency**

* The Operator is effectively a container distributed through the OCI Data Science platform. When deployed through Jobs or Model Deployment, customers can scale up the compute shape, memory size, and load balancer to make the prediciton progressively faster. Please consult an OCI Data Science Platform expert for more specifc advice.
