The ``properties`` is an instance of the ``ModelProperties`` class and has the following predefined fields:

- ``bucket_uri`` (str):
- ``compartment_id`` (str):
- ``deployment_access_log_id`` (str):
- ``deployment_bandwidth_mbps`` (int):
- ``deployment_instance_count`` (int):
- ``deployment_instance_shape`` (str):
- ``deployment_log_group_id`` (str):
- ``deployment_predict_log_id`` (str):
- ``inference_conda_env`` (str):
- ``inference_python_version`` (str):
- ``overwrite_existing_artifact`` (bool):
- ``project_id`` (str):
- ``remove_existing_artifact`` (bool):
- ``training_conda_env`` (str):
- ``training_id`` (str):
- ``training_python_version`` (str):
- ``training_resource_id`` (str):
- ``training_script_path`` (str):

By default, ``properties`` is populated from the environment variables when not specified. For example, in notebook sessions the environment variables are preset and stored in project id  (``PROJECT_OCID``) and compartment id (``NB_SESSION_COMPARTMENT_OCID). So ``properties`` populates these environment variables, and uses the values in methods such as ``.save()`` and ``.deploy()``. Pass in values to overwrite the defaults.  When you use a method that includes an instance of  ``properties``, then ``properties`` records the values that you pass in.  For example, when you pass ``inference_conda_env`` into the ``.prepare()`` method, then ``properties`` records the value.  To reuse the properties file in different places, you can export the properties file using the ``.to_yaml()`` method then reload it into a different machine using the ``.from_yaml()`` method.