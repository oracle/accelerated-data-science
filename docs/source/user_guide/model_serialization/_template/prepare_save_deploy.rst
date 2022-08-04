.. versionadded:: 2.6.3

The ``.prepare_save_deploy()`` method is a shortcut for the functions ``.prepare()``, ``.save()``, and ``.deploy()``. This method returns a ``ModelDeployment`` object and is available for all frameworks. The method takes the following parameters:

* ``inference_conda_env``: (str, optional). Defaults to None.
    Can either be a slug or an object storage path for the conda pack.
    You can only pass in slugs if the conda pack is a service pack.
* ``inference_python_version``: (str, optional). Defaults to None.
    The Python version to use in the deployment.
* ``training_conda_env``: (str, optional). Defaults to None.
    Can either be a slug or an object storage path for the conda pack.
    You can only pass in slugs if the conda pack is a service pack.
* ``training_python_version``: (str, optional). Defaults to None.
    Python version to use for training.
* ``model_file_name``: (str).
    Name of the serialized model.
* ``as_onnx``: (bool, optional). Defaults to False.
    Whether to serialize as ONNX model.
* ``initial_types``: (list[Tuple], optional).
    Defaults to None. Only used for SklearnModel, LightGBMModel and XGBoostModel.
    Each element is a tuple of a variable name and a type.
    Check this link :ref:`http://onnx.ai/sklearn-onnx/api_summary.html#id2>` for
    explanations and examples for ``initial_types``.
* ``force_overwrite``: (bool, optional). Defaults to False.
    Whether to overwrite existing files.
* ``namespace``: (str, optional).
    Namespace of region. Use this parameter to identify the service pack region
    when you pass a slug to ``inference_conda_env`` and ``training_conda_env``.
* ``use_case_type``: str
    The use case type of the model. Assign a value using the``UseCaseType`` class or provide a string in ``UseCaseType``. For
    example, use_case_type=UseCaseType.BINARY_CLASSIFICATION or use_case_type="binary_classification". Check
    the ``UseCaseType`` class to see supported types.
* ``X_sample``: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
    A sample of input data used to generate input schema.
* ``y_sample``: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
    A sample of output data used to generate output schema.
* ``training_script_path``: str. Defaults to None.
    Training script path.
* ``training_id``: (str, optional). Defaults to value from environment variables.
    The training OCID for model. Can be notebook session or job OCID.
* ``ignore_pending_changes``: bool. Defaults to False.
    Whether to ignore pending changes in git.
* ``max_col_num``: (int, optional). Defaults to ``utils.DATA_SCHEMA_MAX_COL_NUM``.
    Do not generate the input schema if the input has more than this
    number of features(columns).
* ``model_display_name``: (str, optional). Defaults to None.
    The name of the model.
* ``model_description``: (str, optional). Defaults to None.
    The description of the model.
* ``model_freeform_tags`` : Dict(str, str), Defaults to None.
    Freeform tags for the model.
* ``model_defined_tags`` : (Dict(str, dict(str, object)), optional). Defaults to None.
    Defined tags for the model.
* ``ignore_introspection``: (bool, optional). Defaults to None.
    Determine whether to ignore the result of model introspection or not.
    If set to True, the save will ignore all model introspection errors.
* ``wait_for_completion`` : (bool, optional). Defaults to True.
    Determine whether to wait for deployment to complete before proceeding.
* ``display_name``: (str, optional). Defaults to None.
    The name of the model.
* ``description``: (str, optional). Defaults to None.
    The description of the model.
* ``deployment_instance_shape``: (str, optional). Default to ``VM.Standard2.1``.
    The shape of the instance used for deployment.
* ``deployment_instance_count``: (int, optional). Defaults to 1.
    The number of instances used for deployment.
* ``deployment_bandwidth_mbps``: (int, optional). Defaults to 10.
    The bandwidth limit on the load balancer in Mbps.
* ``deployment_log_group_id``: (str, optional). Defaults to None.
    The oci logging group id. The access log and predict log share the same log group.
* ``deployment_access_log_id``: (str, optional). Defaults to None.
    The access log OCID for the access logs. :ref:`https://docs.oracle.com/iaas/data-science/using/model_dep_using_logging.htm>`
* ``deployment_predict_log_id``: (str, optional). Defaults to None.
    The predict log OCID for the predict logs. :ref:`https://docs.oracle.com/iaas/data-science/using/model_dep_using_logging.htm>`
* ``kwargs``:
    * ``impute_values``: (dict, optional).
        The dictionary where the key is the column index (or names is accepted
        for pandas dataframe) and the value is the impute value for the corresponding column.
    * ``project_id``: (str, optional).
        Project OCID. If not specified, gets the value either
        from the environment variables or model properties.
    * ``compartment_id`` : (str, optional).
        Compartment OCID. If not specified, gets the value either
        from the environment variables or model properties.
    * ``timeout``: (int, optional). Defaults to 10 seconds.
        The connection timeout in seconds for the client.
    * ``max_wait_time`` : (int, optional). Defaults to 1200 seconds.
        Maximum amount of time to wait in seconds.
        Negative values imply infinite wait time.
    * ``poll_interval`` : (int, optional). Defaults to 60 seconds.
        Poll interval in seconds.
