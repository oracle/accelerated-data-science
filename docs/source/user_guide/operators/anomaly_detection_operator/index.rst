=================
Anomaly Detection
=================

The Anomaly Detection Operator is a low code tool for integrating Anomaly Detection into any enterprise applicaiton. Specifically, it leverages timeseries constructive anomaly detection in order to flag anomolous moments in your data, by time and by ID.


Overview
--------

**Input Data**

The Anomaly Detection Operator accepts a dataset with:
# A datetime column 
# A target column
# (Optionally) 1 or more seires columns (such that the target is indexed by datetime and series)
# (Optionall) An arbitrary number of additional variables

Besides this input data, the user can also specify validation data, if available. Validation data should have all the columns of the input data plus a binary column titled "anomaly". The "anomaly" column should be -1 for anomalies and 1 for normal rows.

Finally the user can provide "test_data" in order to recieve test metrics and evaluate the Operator's performance more easily. Test data should indexed by date and (optionally) series. Test data should have a -1 for anomalous rows and 1 for normal rows.

**Multivariate vs. Univariate Anomaly Detection**

If you have additional variables that you think might be related, then you should use "multivariate" AD. All additional columns given in the input data will be used in determining if the target column is anomalous.

**Auto Model Selection**

Operators users don't need to know anything about the underlying models in order to use them. By default we set ``model: auto``. However, some users want more control over the modeling parameters. These users can set the ``model`` parameter to either ``autots`` or ``automlx`` and then pass parameters directly into ``model_kwargs``. See :doc:`Advanced Examples <./advanced_use_cases>`

**Anomaly Detection Documentation**

This documentation will explore these concepts in greater depth, demonstrating how to leverage the flexibility and configurability of the Python library module to implement both multivariate and univariate anomaly detection models. By the end of this guide, users will have the knowledge and tools needed to make informed decisions when designing anomaly detection solutions tailored to their specific requirements.

.. versionadded:: 2.10.1

.. toctree::
  :maxdepth: 1

  ./quickstart
  ./use_cases
  ./install
  ./productionize
  ./advanced_use_cases
  ./yaml_schema
  ./faq
