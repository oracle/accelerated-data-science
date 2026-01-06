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



.. list-table:: Forecast Operator Configuration Reference
   :widths: 20 10 10 20 40
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Default
     - Description

   * - historical_data
     - dict
     - Yes
     - {"url": "data.csv"}
     - Indexed by date and optionally target category. Includes targets and endogeneous data.

   * - additional_data
     - dict
     - No
     -
     - Optional exogeneous data. Must align with historical_data structure.

   * - test_data
     - dict
     - No
     -
     - Optional, used for evaluation if provided.

   * - output_directory
     - dict
     - No
     -
     - Where output files will be saved. Accepts the same data schema as inputs.

   * - report_filename
     - string
     - No
     - report.html
     - Output report file name.

   * - report_title
     - string
     - No
     -
     - Title of the output report.

   * - report_theme
     - string
     - No
     - light
     - Theme of the report. Options: light, dark.

   * - metrics_filename
     - string
     - No
     - metrics.csv
     - Filename for metrics output.

   * - test_metrics_filename
     - string
     - No
     - test_metrics.csv
     - Filename for test set evaluation metrics.

   * - forecast_filename
     - string
     - No
     - forecast.csv
     - Output forecast data file.

   * - global_explanation_filename
     - string
     - No
     - global_explanations.csv
     - File for global explanations.

   * - local_explanation_filename
     - string
     - No
     - local_explanations.csv
     - File for local explanations.

   * - target_column
     - string
     - Yes
     - target
     - Column to forecast.

   * - datetime_column.name
     - string
     - Yes
     - Date
     - Timestamp column name.

   * - datetime_column.format
     - string
     - No
     -
     - Optional datetime format.

   * - target_category_columns
     - list
     - No
     - ["Series ID"]
     - Categories for multi-series forecasting.

   * - horizon
     - integer
     - Yes
     - 1
     - Forecast horizon (how far ahead).

   * - model
     - string
     - No
     - prophet
     - Model to use. Options: prophet, arima, neuralprophet, theta, automlx, autots, auto-select.

   * - model_kwargs
     - dict
     - No
     -
     - Parameters specific to the chosen model.

   * - preprocessing.enabled
     - boolean
     - No
     - true
     - Whether to apply preprocessing.

   * - preprocessing.steps.missing_value_imputation
     - boolean
     - No
     - true
     - Impute missing values.

   * - preprocessing.steps.outlier_treatment
     - boolean
     - No
     - false
     - Handle outliers.

   * - generate_explanations
     - boolean
     - No
     - false
     - Toggle local and global explanations.

   * - explanations_accuracy_mode
     - string
     - No
     - FAST_APPROXIMATE
     - Explanation mode. Options: HIGH_ACCURACY, BALANCED, FAST_APPROXIMATE, AUTOMLX.

   * - generate_report
     - boolean
     - No
     - true
     - Enable report generation.

   * - generate_metrics
     - boolean
     - No
     - true
     - Enable metrics file generation.

   * - metric
     - string
     - No
     - MAPE
     - Evaluation metric. Options: MAPE, RMSE, MSE, SMAPE (case-insensitive).

   * - what_if_analysis
     - dict
     - No
     -
     - Save models to model catalog if enabled. Includes deployment config.

   * - previous_output_dir
     - string
     - No
     -
     - Load previous run outputs.

   * - generate_model_parameters
     - boolean
     - No
     -
     - Export fitted model parameters.

   * - generate_model_pickle
     - boolean
     - No
     -
     - Export trained model as pickle file.

   * - confidence_interval_width
     - float
     - No
     - 0.80
     - Width of confidence intervals in forecast.

   * - tuning.n_trials
     - integer
     - No
     - 10
     - Number of tuning trials for hyperparameter search.


Further Description
-------------------


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

    * **model**: (Optional) The name of the model framework to use. Defaults to ``auto-select``. Available options include ``arima``, ``prophet``, ``neuralprophet``, ``theta``, ``autots``, and ``auto-select``.

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

    * **what_if_analysis**: (Optional) This dictionary defines the configuration for saving the model to the model store and setting up a model deployment server to enable real-time predictions and what-if analysis, with the following parameters:
        * **project_id**: The OCID of the data science project where the resources will be created.
        * **compartment_id**: The OCID of the compartment
        * **model_display_name**: The display name of the model used to save the model in the model store.
        * **model_deployment**: This dictionary describing the model deployment configuration. It includes:
            * **display_name**: The display name for the model deployment.
            * **initial_shape**: The compute shape for the initial model deployment.
            * **description**: A brief description of the model deployment.
            * **log_group**: The OCID of the log group where the logs are organized.
            * **log_id**: The OCID of the log where deployment logs are stored.
            * **auto_scaling**: (Optional) A dictionary specifying the auto-scaling configuration for the deployment. It includes:
                * **minimum_instance**: The minimum number of instances to maintain during auto-scaling.
                * **maximum_instance**: The maximum number of instances to scale up to during peak demand.
                * **cool_down_in_seconds**: The cooldown period (in seconds) to wait before performing another scaling action.
                * **scaling_metric**: The metric used for scaling actions. e.g. ``CPU_UTILIZATION`` or  ``MEMORY_UTILIZATION``
                * **scale_in_threshold**: The utilization percentage below which the instances will scale in (reduce).
                * **scale_out_threshold**: The utilization percentage above which the instances will scale out (increase).
