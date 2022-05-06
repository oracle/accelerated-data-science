The ``properties`` is an instance of the ``ModelProperties`` class and has the following predefined fields:

- ``compartment_id: str``
- ``deployment_access_log_id: str``
- ``deployment_bandwidth_mbps: int``
- ``deployment_instance_count: int``
- ``deployment_instance_shape: str``
- ``deployment_log_group_id: str``
- ``deployment_predict_log_id: str``
- ``inference_conda_env: str``
- ``inference_python_version: str``
- ``project_id: str``
- ``training_conda_env: str``
- ``training_id: str``
- ``training_python_version: str``
- ``training_resource_id: str``
- ``training_script_path: str``

By default, ``properties`` is populated from the appropriate environment variables if it's
not specified. For example, in a notebook session, the environment variables
for project id and compartment id are preset and stored in ``PROJECT_OCID`` and
``NB_SESSION_COMPARTMENT_OCID`` by default. So ``properties`` populates these variables 
from the environment variables and uses the values in methods such as ``.save()`` and ``.deploy()``.
However, you can explicitly pass in values to overwrite the defaults.
When you use a method that includes an instance of  ``properties``, then ``properties`` records the values that you pass in.
For example, when you pass ``inference_conda_env`` into the ``.prepare()`` method, then ``properties`` records this value.
To reuse the properties file in different places, you can export the properties file using the ``.to_yaml()`` method and reload it into a different machine using the ``.from_yaml()`` method.
