The ``.prepare()`` method serializes the model and prepares and saves the ``score.py`` and ``runtime.yaml`` files using the following parameters:
  
- ``as_onnx: (bool, optional)``: Defaults to ``False``. If ``True``, it will serialize as an ONNX model.
- ``force_overwrite: (bool, optional)``: Defaults to ``False``. If ``True``, it will overwrite existing files.
- ``ignore_pending_changes: bool``: Defaults to ``False``. If ``False``, it will ignore the pending changes in Git.
- ``inference_conda_env: (str, optional)``: Defaults to ``None``. Can be either slug or the Object Storage path of the conda environment. You can only pass in slugs if the conda environment is a Data Science service environment.
- ``inference_python_version: (str, optional)``: Defaults to ``None``. The version of Python to use in the model deployment.
- ``max_col_num: (int, optional)``: Defaults to ``utils.DATA_SCHEMA_MAX_COL_NUM``. Do not automatically generate the input schema if the input data has more than this number of features.
- ``model_file_name: (str)``: Name of the serialized model.
- ``namespace: (str, optional)``: Namespace of the OCI region. This is used for identifying which region the service environment is from when you provide a slug to the ``inference_conda_env`` or ``training_conda_env`` parameters.
- ``training_conda_env: (str, optional)``: Defaults to ``None``. Can be either slug or object storage path of the conda environment that was used to train the model. You can only pass in a slug if the conda environment is a Data Science service environment.
- ``training_id: (str, optional)``: Defaults to value from environment variables. The training OCID for the model. Can be a notebook session or job OCID.
- ``training_python_version: (str, optional)``: Defaults to None. The version of Python used to train the model.
- ``training_script_path: str``: Defaults to ``None``. The training script path.
- ``use_case_type: str``: The use case type of the model. Use it with the ``UserCaseType`` class or the string provided in ``UseCaseType``. For example, ``use_case_type=UseCaseType.BINARY_CLASSIFICATION`` or ``use_case_type="binary_classification"``, see the ``UseCaseType`` class to see all supported types.
- ``X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]``: Defaults to ``None``. A sample of the input data. It is used to generate the input schema.
- ``y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]``: Defaults to None. A sample of output data. It is used to generate the output schema.
- ``**kwargs``:
    - ``impute_values: (dict, optional)``: The dictionary where the key is the column index (or names is accepted for Pandas dataframe), and the value is the imputed value for the corresponding column.

