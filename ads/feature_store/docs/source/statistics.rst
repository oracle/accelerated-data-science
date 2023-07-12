.. _Statistics:

Statistics
*************

Feature Store provides functionality to compute statistics for feature groups as well as datasets and persist them along with the metadata. These statistics can help you
to derive insights about the data quality. These statistical metrics are computed during materialisation time and persisting with other metadata.

.. note::

  Feature Store utilizes MLM Insights which is a Python API that helps evaluate and monitor data for entirety of ML Observability lifecycle. It performs data summarization which reduces a dataset into a set of descriptive statistics.

The statistical metrics that are computed by feature store depend on the feature type.

+------------------------+-----------------------+
| Numerical Metrics      | Categorical Metrics   |
+========================+=======================+
| Skewness               | Count                 |
+------------------------+-----------------------+
| StandardDeviation      | TopKFrequentElements  |
+------------------------+-----------------------+
| Min                    | TypeMetric            |
+------------------------+-----------------------+
| IsConstantFeature      | DuplicateCount        |
+------------------------+-----------------------+
| IQR                    | Mode                  |
+------------------------+-----------------------+
| Range                  | DistinctCount         |
+------------------------+-----------------------+
| ProbabilityDistribution|                       |
+------------------------+-----------------------+
| Variance               |                       |
+------------------------+-----------------------+
| FrequencyDistribution  |                       |
+------------------------+-----------------------+
| Count                  |                       |
+------------------------+-----------------------+
| Max                    |                       |
+------------------------+-----------------------+
| DistinctCount          |                       |
+------------------------+-----------------------+
| Sum                    |                       |
+------------------------+-----------------------+
| IsQuasiConstantFeature |                       |
+------------------------+-----------------------+
| Quartiles              |                       |
+------------------------+-----------------------+
| Mean                   |                       |
+------------------------+-----------------------+
| Kurtosis               |                       |
+------------------------+-----------------------+
