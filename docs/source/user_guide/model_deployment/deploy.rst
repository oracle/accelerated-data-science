Deploy
******

The ``.deploy()`` method of the ``ModelDeployment`` class is used to create a model deployment.  It has the following parameters:

  * ``max_wait_time``: The timeout limit, in seconds, for the deployment process to wait until it is active. Defaults to 1200 seconds.
  * ``poll_interval``: The interval between checks of the deployment status in seconds. Defaults to 30 seconds.
  * ``wait_for_completion``: Blocked process until the deployment has been completed. Defaults to ``True``.

To deploy a ``ModelDeployment``, you need to define a ``ModelDeployment`` object first and then call ``.deploy()``. There are two ways to define a ``ModelDeployment`` object.

Builder Pattern
===============

Infrastructure
--------------

You define the model deployment infrastructure by passing the following properties to ``ModelDeploymentInfrastructure``:

  * ``access_log``: Log group OCID and log OCID for the access logs.
  * ``bandwidth_mbps``: The bandwidth limit on the load balancer in Mbps.
  * ``compartment_id``: Compartment OCID that the model deployment belongs to.
  * ``replica``: The number of instances to deploy.
  * ``shape_name``: The instance shape name to use. For example, "VM.Standard.E4.Flex".
  * ``shape_config_details``: The instance shape configure details to use if flexible shape is selected for ``shape_name``. 
  * ``predict_log``: Log group OCID and log OCID for the predict logs.
  * ``project_id``: Project OCID that the model deployment will belong to.
  * ``web_concurrency``: The web concurrency to use. 

Below is an example to define a ``ModelDeploymentInfrastructure`` object

.. code-block:: python3

    from ads.model.deployment.model_deployment_infrastructure import ModelDeploymentInfrastructure

    infrastructure = (
        ModelDeploymentInfrastructure()
        .with_project_id("<PROJECT_OCID>")
        .with_compartment_id("<COMPARTMENT_OCID>")    
        .with_shape_name("VM.Standard.E4.Flex")
        .with_shape_config_details(
            ocpus=1,
            memory_in_gbs=16
        )
        .with_replica(1)
        .with_bandwidth_mbps(10)
        .with_web_concurrency(10)
        .with_access_log(
            log_group_id="<ACCESS_LOG_GROUP_OCID>", 
            log_id="<ACCESS_LOG_OCID>"
        )
        .with_predict_log(
            log_group_id="<PREDICT_LOG_GROUP_OCID>", 
            log_id="<PREDICT_LOG_OCID>"
        )
    )


Runtime
-------

The Data Science Model Deployment supports service managed conda runtime and customized docker container runtime.

  * ``ModelDeploymentContainerRuntime`` allows you to deploy model deployment on customized docker container runtime.
  * ``ModelDeploymentCondaRuntime`` allows you to deploy model deployment on service-managed conda runtime.

ModelDeploymentContainerRuntime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the ``ModelDeploymentContainerRuntime``, you need to first push the image to `OCI container registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_. See `Creating a Repository <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ and `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ for more details.
 
You can define the model deployment container runtime by passing the following properties to ``ModelDeploymentContainerRuntime`` object:

  * ``model_uri``: The model ocid or path to model artifacts directory that is used in the model deployment.
  * ``deployment_mode``: The mode of model deployment. Allowed deployment modes are ``HTTPS_ONLY`` and ``STREAM_ONLY``. Optional.
  * ``input_stream_ids``: The input stream ids for model deployment. Required when deployment mode is ``STREAM_ONLY``.
  * ``output_stream_ids``: The output stream ids for model deployment. Required when deployment mode is ``STREAM_ONLY``.
  * ``env``: The environment variables. Optional.
  * ``image``: The full path of docker container image to the OCIR registry. The acceptable formats are: ``<region>.ocir.io/<registry>/<image>:<tag>`` and ``<region>.ocir.io/<registry>/<image>:<tag>@digest``. Required.
  * ``image_digest``: The docker container image digest. Optional.
  * ``entrypoint``: The entrypoint to docker container image. Optional.
  * ``server_port``: The server port of docker container image. Optional.
  * ``health_check_port``: The health check port of docker container image. Optional.
  * ``cmd``: The additional commands to docker container image. The commands can be args to the entrypoint or the only command to execute in the absence of an entrypoint. Optional.

Below is an example to define a ``ModelDeploymentContainerRuntime`` object

.. code-block:: python3

    from ads.model.deployment.model_deployment_runtime import ModelDeploymentContainerRuntime

    container_runtime = (
        ModelDeploymentContainerRuntime()
        .with_image("<IMAGE_PATH_TO_OCIR>")
        .with_image_digest("<IMAGE_DIGEST>")
        .with_entrypoint(["python","/opt/ds/model/deployed_model/api.py"])
        .with_server_port(5000)
        .with_health_check_port(5000)
        .with_env({"key":"value"})
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_uri("<MODEL_URI>")
    )


ModelDeploymentCondaRuntime
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define the model deployment conda runtime by passing the following properties to ``ModelDeploymentCondaRuntime`` object:

  * ``model_uri``: The model ocid or path to model artifacts that is used in the model deployment.
  * ``deployment_mode``: The deployment mode. The allowed deployment modes are ``HTTPS_ONLY`` and ``STREAM_ONLY``. Optional.
  * ``input_stream_ids``: The input stream ids for model deployment. Required when deployment mode is ``STREAM_ONLY``.
  * ``output_stream_ids``: The output stream ids for model deployment. Required when deployment mode is ``STREAM_ONLY``.
  * ``env``: The environment variables. Optional.

Below is an example to define a ``ModelDeploymentCondaRuntime`` object

.. code-block:: python3

    from ads.model.deployment.model_deployment_runtime import ModelDeploymentCondaRuntime

    conda_runtime = (
        ModelDeploymentCondaRuntime()
        .with_env({"key":"value"})
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_uri("<MODEL_URI>")
    )


ModelDeployment
~~~~~~~~~~~~~~~

You can define the model deployment by passing the following properties to ``ModelDeployment`` object:

  * ``defined_tags``: A dictionary of defined tags to be attached to the model deployment. Optional.
  * ``description``: A description of the model deployment. Optional.
  * ``display_name``: A name that identifies the model deployment in the Console.
  * ``freeform_tags``: A dictionary of freeform tags to be attached to the model deployment. Optional.
  * ``runtime``: The runtime configuration to be attached to the model deployment.
  * ``infrastructure``: The infrastructure configuration to be attached to the model deployment.

Below is an example to define and deploy a ``ModelDeployment`` object with custom docker container runtime

.. code-block:: python3

    from ads.model.deployment import ModelDeployment

    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags({"key1":"value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(container_runtime)
    )

    deployment.deploy(wait_for_completion=False)


YAML Serialization
==================

A ``ModelDeployment`` object can be serialized to a YAML file by calling ``to_yaml()``, which returns the YAML as a string.  You can easily share the YAML with others, and reload the configurations by calling ``from_yaml()``.  The ``to_yaml()`` and ``from_yaml()`` methods also take an optional ``uri`` argument for saving and loading the YAML file.  This argument can be any URI to the file location supported by `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__, including Object Storage. For example:

.. code-block:: python3

    # Save the model deployment configurations to YAML file
    deployment.to_yaml(uri="oci://bucket_name@namespace/path/to/deployment.yaml")

    # Load the model deployment configurations from YAML file
    deployment = ModelDeployment.from_yaml(uri="oci://bucket_name@namespace/path/to/deployment.yaml")

    # Save the model deployment configurations to YAML in a string
    yaml_string = ModelDeployment.to_yaml()

    # Load the model deployment configurations from a YAML string
    deployment = ModelDeployment.from_yaml("""
    kind: deployment
    spec:
        infrastructure:
        kind: infrastructure
            ...
    """")

    deployment.deploy(wait_for_completion=False)

Here is an example of a YAML file representing the ``ModelDeployment`` with docker container runtime defined in the preceding examples:


.. code-block:: yaml

    kind: deployment
    spec:
      displayName: Model Deployment Demo using ADS
      description: The model deployment description
      freeform_tags:
        key1: value1
      infrastructure:
        kind: infrastructure
        type: datascienceModelDeployment
        spec:
          compartmentId: <COMPARTMENT_OCID>
          projectId: <PROJECT_OCID>
          accessLog:
            logGroupId: <ACCESS_LOG_GROUP_OCID>
            logId: <ACCESS_LOG_OCID>
          predictLog:
            logGroupId: <PREDICT_LOG_GROUP_OCID>
            logId: <PREDICT_LOG_OCID>
          shapeName: VM.Standard.E4.Flex
          shapeConfigDetails:
            memoryInGBs: 16
            ocpus: 1
          replica: 1
          bandWidthMbps: 10
      runtime:
        kind: runtime
        type: container
        spec:
          modelUri: <MODEL_URI>
          image: <IMAGE_PATH_TO_OCIR>
          imageDigest: <IMAGE_DIGEST>
          entrypoint: ["python","/opt/ds/model/deployed_model/api.py"]
          serverPort: 5000
          healthCheckPort: 5000
          env:
            WEB_CONCURRENCY: "10"
          deploymentMode: HTTPS_ONLY


**ADS ModelDeployment YAML schema**

.. code-block:: yaml

    kind:
      required: true
      type: string
      allowed:
        - deployment
    spec:
      required: true
      type: dict
      schema:
        displayName:
        type: string
        required: false
      description:
        type: string
        required: false
      freeform_tags:
        type: dict
        required: false
      defined_tags:
        type: dict
        required: false
      infrastructure:
        type: dict
        required: true
      runtime:
        type: dict
        required: true

**ADS Model Deployment Infrastructure YAML Schema**

.. code-block:: yaml

    kind:
      required: true
      type: string
      allowed:
        - infrastructure
    type:
      required: true
      type: string
      allowed:
        - datascienceModelDeployment
    spec:
      compartmentId:
        type: string
        required: true
      projectId:
        type: string
        required: true
      bandWidthMbps:
        type: integer
        required: false
      webConcurrency:
        type: integer
        required: false
      logGroupId:
        type: string
        required: false
      logId:
        type: string
        required: false
      accessLog:
        type: dict
        nullable: true
        required: false
        schema:
          logId:
            required: false
            type: string
          logGroupId:
            required: false
            type: string
      predictLog:
        type: dict
        nullable: true
        required: false
        schema:
          logId:
            required: false
            type: string
          logGroupId:
            required: false
            type: string 
      shapeName:
        type: string
        required: false
      shapeConfigDetails:
        type: dict
        nullable: true
        required: false
        schema:
          ocpus:
            required: true
            type: float
          memoryInGBs:
            required: true
            type: float  
      replica:
        type: integer
        required: false

**ADS Model Deployment Conda Runtime YAML Schema**

.. code-block:: yaml
    
    kind:
      required: true
      type: string
      allowed: 
        - runtime
    type: 
      required: true
      type: string
      allowed:
        - conda 
    spec:
      modelUri:
        type: string
        required: true
      env:
        type: dict
        required: false
      inputStreamIds: 
        type: list
        required: false
      outputStreamIds:
        type: list
        required: false
      deploymentMode:
        type: string
        required: false

**ADS Model Deployment Container Runtime YAML Schema**

.. code-block:: yaml
    
    kind:
      required: true
      type: string
      allowed: 
        - runtime
    type: 
      required: true
      type: string
      allowed:
        - container 
    spec:
      modelUri:
        type: string
        required: true
      image:
        type: string
        required: true
      imageDigest:
        type: string
        required: false
      entrypoint:
        type: list
        required: false
      cmd:
        type: list
        required: false
      serverPort:
        type: integer
        required: false
      healthCheckPort:
        type: integer
        required: false
      env:
        type: dict
        required: false
      inputStreamIds: 
        type: list
        required: false
      outputStreamIds:
        type: list
        required: false
      deploymentMode:
        type: string
        required: false

