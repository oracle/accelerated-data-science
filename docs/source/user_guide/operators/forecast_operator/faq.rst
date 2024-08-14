====
FAQs
====

**How do I learn more about AutoMLX?**

More details in the documentation here: https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/latest/automl.html

**How do I learn More about AutoTS?**

More details in the documentation here: https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html

**Pip Install Failing with "ERROR: No matching distribution found for oracle-automlx==23.4.1; extra == "forecast""**

Automlx only supports Python<=3.8, != 3.9, <= 3.10.7 . If you are builing in Python 3.9 or 3.10.8+, no automlx distribution will be available. It's recommended to use the conda pack available through Notebook Sessions, or to use Python 3.8

**How do you impute missing values?**

We use linear interpolation by default

**Is there a way to specify the % increase which should be marked as anomaly?**
Yes, we can use the contamination parameter to control the percentage of anomalies. The default value is 0.1

**How is seasonality handled?**
Seasonality is examined differently by each modeling framework. Refer to a specific model's documentation for more information.
