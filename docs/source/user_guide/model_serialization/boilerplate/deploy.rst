You can use the ``.deploy()`` method to deploy a model. You must first save the model to the model catalog, and then deploy it. 

The ``.deploy()`` method returns a ``ModelDeployment`` object.  Specify deployment attributes such as display name, instance type, number of instances,  maximum router bandwidth, and logging groups.  The API takes the following parameters:

- ``deployment_access_log_id: (str, optional)``: Defaults to ``None``. The access log OCID for the access logs, see `logging <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm>`_.
- ``deployment_bandwidth_mbps: (int, optional)``: Defaults to 10. The bandwidth limit on the load balancer in Mbps.
- ``deployment_instance_count: (int, optional)``: Defaults to 1. The number of instances used for deployment.
- ``deployment_instance_shape: (str, optional)``: Default to VM.Standard2.1. The shape of the instance used for deployment.
- ``deployment_log_group_id: (str, optional)``: Defaults to ``None``. The OCI logging group OCID. The access log and predict log share the same log group.
- ``deployment_predict_log_id: (str, optional)``: Defaults to ``None``. The predict log OCID for the predict logs, see `logging <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm>`_.
- ``description: (str, optional)``: Defaults to ``None``. The description of the model.
- ``display_name: (str, optional)``: Defaults to ``None``. The name of the model.
- ``wait_for_completion : (bool, optional)``: Defaults to ``True``. Set to wait for the deployment to complete before proceeding.
- ``**kwargs``:
    - ``compartment_id : (str, optional)``: Compartment OCID. If not specified, the value is taken from the environment variables.
    - ``max_wait_time : (int, optional)``: Defaults to 1200 seconds. The maximum amount of time to wait in seconds. A negative value implies an infinite wait time.
    - ``poll_interval : (int, optional)``: Defaults to 60 seconds. Poll interval in seconds.
    - ``project_id: (str, optional)``: Project OCID. If not specified, the value is taken from the environment variables.

