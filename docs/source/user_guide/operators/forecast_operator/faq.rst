====
FAQs
====

**How can I learn more about AutoMLX?**

For more details, refer to the official documentation: `AutoMLX Documentation <https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/latest/automl.html>`_

**How can I learn more about AutoTS?**

For more details, refer to the official documentation: `AutoTS Documentation <https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html>`_

**Why is my pip install failing with "ERROR: No matching distribution found for oracle-automlx==23.4.1; extra == 'forecast'"?**

AutoMLX supports Python versions <=3.8, !=3.9, <=3.10.7. If you are using Python 3.9 or 3.10.8+, no AutoMLX distribution will be available. We recommend using the Conda pack available through Notebook Sessions or downgrading to Python 3.8.

**How do you handle missing values?**

By default, missing values are imputed using linear interpolation.

**Is there a way to specify the percentage increase that should be marked as an anomaly?**

Yes, the ``contamination`` parameter can be used to control the percentage of anomalies. The default value is 0.1 (10%).

**How is seasonality handled?**

Seasonality is analyzed differently by each modeling framework. Refer to the specific model's documentation for more detailed information.
