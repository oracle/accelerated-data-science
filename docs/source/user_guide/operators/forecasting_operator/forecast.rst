===================
Configure Forecast
===================

Let's explore each line of the forecast.yaml so we can better understand options for extending and customizing the operator to our use case.

Here is an example forecast.yaml wit every parameter specified:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: Date
        historical_data:
            url: data.csv
        horizon: 3
        model: auto
        target_column: target


* **Kind**: The yaml file always starts with ``kind: operator``. There are many other kinds of yaml files that can be run by ``ads opctl``, so we need to specify this is an operator.
* **Type**: The type of operator is ``forecast``. 
* **Version**: The only available version is ``v1``.
* **Spec**: Spec contains the bulk of the information for the specific problem.
    * **historical_data**: This dictionary contains the details for how to read the historical data. Historical data must contain the target column, the datetime column, and optionally the target category column.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **target_column**: This string specifies the name of the column where the target data is within the historical data.
    * **datetime_column**: The dictionary outlining details around the datetime column.
        * **name**: the name of the datetime column. Must be the same in both historical and additional data.
        * **format**: the format of the datetime string in python notation `detailed here <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
    * **horizon**: the integer number of periods to forecast.

    * **target_category_columns**: (optional) The category ID of the target. 
    * **additional_data**: (optional) This dictionary contains the details for how to read the addtional data. Additional data must contain the the datetime column, the target category column (if present in historical), and any other columns with values over the horizon.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **output_directory**: (optional) This dictionary contains the details for where to put the output artifacts. The directory need not exist, but must be accessible by the Operator during runtime.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **model**: (optional) The name of the model framework you want to use. Defaults to "auto". Other options are: ``arima``, ``automlx``, ``prophet``, ``neuralprophet``, ``autots``, and ``auto``.
    * **model_kwargs**: (optional) This kwargs dict passes straight through to the model framework.
    * **test_data**: (optional) This dictionary contains the details for how to read the test data. Test data must be formatted identically to historical data and contain values for every period in the forecast horizon.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.

    * **tuning**: (optional) This dictionary specific details around tuning the NeuralProphet and Prophet models.
        * **n_trials**: The number of separate tuning jobs to run. Increasing this integer increases the time to completion, but may improve the quality.
    * **preprocessing**: (optional) Preprocessing and feature engineering can be disabled using this flag, Defaults to true
    * **metric**: (optional) The metric to select across. Users can select among: MAPE, RMSE, MSE, and SMAPE
    * **confidence_interval_width**: (optional) The width of the confidence interval to caluclate in the forecast and report.html. Defaults to 0.80 meaning an 80% confidence interval   

    * **report_filename**: (optional) Placed into output_directory location. Defaults to report.html
    * **report_title**: (optional) The title of the report.
    * **report_theme**: (optional) Can be "dark" or "light". Defaults to "light".
    * **metrics_filename**: (optional) Placed into output_directory location. Defaults to metrics.csv
    * **test_metrics_filename**: (optional) Placed into output_directory location. Defaults to test_metrics.csv
    * **forecast_filename**: (optional) Placed into output_directory location. Defaults to forecast.csv
    * **generate_explanations**: (optional) Explainability, both local and global, can be disabled using this flag. Defaults to false.
    * **generate_report**: (optional) Report file generation can be enabled using this flag. Defaults to true.
    * **generate_metrics**: (optional) Metrics files generation can be enabled using this flag. Defaults to true.
