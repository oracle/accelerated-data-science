===========
YAML Schema
===========

In this document, we'll explore each line of the ``forecast.yaml`` file to better understand the options available for extending and customizing the operator for specific use cases.

Below is an example of a ``forecast.yaml`` file with every parameter specified:

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
        target_column: target

* **kind**: The YAML file always starts with ``kind: operator``. This identifies the type of service. Common kinds include ``operator`` and ``job``, but here, ``operator`` is required.
* **type**: The type of operator is ``forecast``, which should always be specified when using this forecast operator.
* **version**: The only available version is ``v1``.
* **spec**: This section contains the main configuration details for the forecasting problem.

    * **historical_data**: This dictionary specifies how to load the historical data, which must include the target column, the datetime column, and optionally, the target category column.
        * **url**: Provide the URI for the dataset, using a pattern like ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **format**: (Optional) Specify the format of the dataset (e.g., ``csv``, ``json``, ``excel``).
        * **options**: (Optional) Include any additional arguments for loading the data, such as ``filters``, ``columns``, and ``sql`` query parameters.
        * **vault_secret_id**: (Optional) The Vault secret ID for secure access if needed.

    * **target_column**: This string specifies the name of the target data column within the historical data. The default is ``target``.
    
    * **datetime_column**: This dictionary outlines details about the datetime column.
        * **name**: The name of the datetime column. It must match between the historical and additional data. The default is ``Date``.
        * **format**: (Optional) Specify the format of the datetime string using Python's ``strftime`` format codes. Refer to the ``datetime`` documentation for details.

    * **horizon**: The number of periods to forecast, specified as an integer. The default value is 1.

    * **target_category_columns**: (Optional) A list of target category columns. The default is ``["Series ID"]``.
    
    * **additional_data**: (Optional) This dictionary specifies how to load additional datasets, which must be indexed by the same targets and categories as the historical data and include data points for each date/category in the forecast horizon.
        * **url**: Provide the URI for the dataset, using a pattern like ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **format**: (Optional) Specify the format of the dataset (e.g., ``csv``, ``json``, ``excel``).
        * **options**: (Optional) Include any additional arguments for loading the data, such as ``filters``, ``columns``, and ``sql`` query parameters.
        * **vault_secret_id**: (Optional) The Vault secret ID for secure access if needed.

    * **output_directory**: (Optional) This dictionary specifies where to save output artifacts. The directory does not need to exist beforehand, but it must be accessible during runtime.
        * **url**: Provide the URI for the output directory, using a pattern like ``oci://<bucket>@<namespace>/subfolder/``.
        * **format**: (Optional) Specify the format for output data (e.g., ``csv``, ``json``, ``excel``).
        * **options**: (Optional) Include any additional arguments, such as connection parameters for storage.

    * **model**: (Optional) The name of the model framework to use. Defaults to ``auto-select``. Available options include ``arima``, ``prophet``, ``neuralprophet``, ``autots``, and ``auto-select``.

    * **model_kwargs**: (Optional) A dictionary of arguments to pass directly to the model framework, allowing for detailed control over modeling.

    * **test_data**: (Optional) This dictionary specifies how to load test data, which must be formatted identically to the historical data and include values for every period in the forecast horizon.
        * **url**: Provide the URI for the dataset, using a pattern like ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **format**: (Optional) Specify the format of the dataset (e.g., ``csv``, ``json``, ``excel``).
        * **options**: (Optional) Include any additional arguments for loading the data, such as ``filters``, ``columns``, and ``sql`` query parameters.
        * **vault_secret_id**: (Optional) The Vault secret ID for secure access if needed.

    * **tuning**: (Optional) This dictionary specifies details for tuning the ``NeuralProphet`` and ``Prophet`` models.
        * **n_trials**: The number of separate tuning jobs to run. Increasing this value may improve model quality but will increase runtime. The default is 10.

    * **preprocessing**: (Optional) Controls preprocessing and feature engineering steps. This can be enabled or disabled using the ``enabled`` flag. The default is ``true``.
        * **steps**: (Optional) Specific preprocessing steps, such as ``missing_value_imputation`` and ``outlier_treatment``, which are enabled by default.

    * **metric**: (Optional) The metric to select during model evaluation. Options include ``MAPE``, ``RMSE``, ``MSE``, and ``SMAPE``. The default is ``MAPE``.

    * **confidence_interval_width**: (Optional) The width of the confidence interval to calculate in the forecast. The default is 0.80, indicating an 80% confidence interval.

    * **report_filename**: (Optional) The name of the report file. It is saved in the output directory, with a default name of ``report.html``.
    
    * **report_title**: (Optional) The title of the report.

    * **report_theme**: (Optional) The visual theme of the report. Options are ``light`` (default) or ``dark``.

    * **metrics_filename**: (Optional) The name of the metrics file. It is saved in the output directory, with a default name of ``metrics.csv``.
    
    * **test_metrics_filename**: (Optional) The name of the test metrics file. It is saved in the output directory, with a default name of ``test_metrics.csv``.
    
    * **forecast_filename**: (Optional) The name of the forecast file. It is saved in the output directory, with a default name of ``forecast.csv``.

    * **generate_explanations**: (Optional) Controls whether to generate explainability reports (both local and global). This feature is disabled by default (``false``).

    * **generate_report**: (Optional) Controls whether to generate a report file. This feature is enabled by default (``true``).

    * **generate_metrics**: (Optional) Controls whether to generate metrics files. This feature is enabled by default (``true``).

    * **global_explanation_filename**: (Optional) The name of the global explanation file. It is saved in the output directory, with a default name of ``global_explanations.csv``.

    * **local_explanation_filename**: (Optional) The name of the local explanation file. It is saved in the output directory, with a default name of ``local_explanations.csv``.
