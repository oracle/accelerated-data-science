===========
YAML Schema
===========

Let's explore each line of the anomaly.yaml so we can better understand options for extending and customizing the operator to our use case.

Here is an example anomaly.yaml with every parameter specified:

.. code-block:: yaml

    kind: operator
    type: anomaly
    version: v1
    spec:
        datetime_column:
            name: Date
        input_data:
            url: data.csv
        model: auto
        target_column: target


* **Kind**: The yaml file always starts with ``kind: operator``. There are many other kinds of yaml files that can be run by ``ads opctl``, so we need to specify this is an operator.
* **Type**: The type of operator is ``anomaly``. 
* **Version**: The only available version is ``v1``.
* **Spec**: Spec contains the bulk of the information for the specific problem.
    * **input_data**: This dictionary contains the details for how to read the input data. Input data must contain the target column, the datetime column, and optionally the target category column.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **target_column**: This string specifies the name of the column where the target data is within the historical data.
    * **datetime_column**: The dictionary outlining details around the datetime column.
        * **name**: the name of the datetime column. Must be the same in both historical and additional data.
        * **format**: the format of the datetime string in python notation `detailed here <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.

    * **target_category_columns**: (optional) The Series ID of the target. When provided, the target data must be present for each date in the datetime_column and for each series id in the target_category_columns.
    * **validation_data**: (optional) This dictionary contains the details for how to read the validation data. Validation data must contain all of the columns of input_data plus a column titles "anomaly".
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **output_directory**: (optional) This dictionary contains the details for where to put the output artifacts. The directory need not exist, but must be accessible by the Operator during runtime.
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/subfolder/``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.
    * **model**: (optional) The name of the model framework you want to use. Defaults to "auto". Other options are: ``arima``, ``automlx``, ``prophet``, ``neuralprophet``, ``autots``, and ``auto``.
    * **model_kwargs**: (optional) This kwargs dict passes straight through to the model framework. If you want to take direct control of the modeling, this is the best way.
    * **test_data**: (optional) This dictionary contains the details for how to read the test data. Test data should contain every datetime value of the input_data, (optionally) all of the series from target_category_columns, and a column titles "anomaly" with either a 1 (non-anomalous) or 0 (anomalous).
        * **url**: Insert the uri for the dataset if it's on object storage or Data Lake using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
        * **kwargs**: Insert any other args for pandas to load the data (``format``, ``options``, etc.) See full list in ``YAML Schema`` section.

    * **preprocessing**: (optional) Preprocessing and feature engineering can be disabled using this flag, Defaults to true

    * **report_filename**: (optional) Placed into output_directory location. Defaults to report.html
    * **report_title**: (optional) The title of the report.
    * **report_theme**: (optional) Can be "dark" or "light". Defaults to "light".
    * **metrics_filename**: (optional) Placed into output_directory location. Defaults to metrics.csv
    * **test_metrics_filename**: (optional) Placed into output_directory location. Defaults to test_metrics.csv
    * **outliers_filename**: (optional) Placed into output_directory location. Defaults to outliers.csv
    * **inliers_filename**: (optional) Placed into output_directory location. Defaults to inliers.csv
    * **generate_report**: (optional) Report file generation can be enabled using this flag. Defaults to true.
    * **generate_metrics**: (optional) Metrics files generation can be enabled using this flag. Defaults to true.
