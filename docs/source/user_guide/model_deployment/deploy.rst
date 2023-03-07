Deploy
******

To deploy a model deployment, you'll need to define a ``ModelDeployment`` object and call the ``.deploy()`` of it. You could either use API or YAML to define the ``ModelDeployment`` object.

Deploy a ModelDeployment on Docker Container Runtime
====================================================

The ADS ``ModelDeploymentContainerRuntime`` class allows you to run a container image using OCI data science model deployment.

To use the ``ModelDeploymentContainerRuntime``, you need to first build a docker container image. See `<build_container_image>` for the end-to-end example. Once you have the image, push it to `OCI container registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_. See `Creating a Repository <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ and `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ for more details.

To configure ``ModelDeploymentContainerRuntime``, you must specify the container ``image``. You can optionally specify the `entrypoint` and `cmd` for running the container (See `Understand how CMD and ENTRYPOINT interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_).

Below is an example of deploying model on docker container runtime:

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.model.deployment.model_deployment_infrastructure import ModelDeploymentInfrastructure
    from ads.model.deployment.model_deployment_runtime import ModelDeploymentContainerRuntime
    from ads.model.deployment import ModelDeployment

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

    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags({"key1":"value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(container_runtime)
    )

    deployment.deploy()

  .. code-tab:: Python3
    :caption: YAML

    from ads.model.deployment import ModelDeployment

    yaml_string = """
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
          webConcurrency: 10
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
    """

    deployment = ModelDeployment.from_yaml(yaml_string)
    deployment.deploy()


Deploy a ModelDeployment on Conda Runtime
=========================================

To deploy a model deployment on conda runtime, you need to configure ``ModelDeploymentCondaRuntime``.

Below is an example of deploying model on conda runtime:

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.model.deployment.model_deployment_infrastructure import ModelDeploymentInfrastructure
    from ads.model.deployment.model_deployment_runtime import ModelDeploymentCondaRuntime
    from ads.model.deployment import ModelDeployment

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

    conda_runtime = (
        ModelDeploymentCondaRuntime()
        .with_env({"key":"value"})
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_uri("<MODEL_URI>")
    )

    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags({"key1":"value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(conda_runtime)
    )

    deployment.deploy()

  .. code-tab:: Python3
    :caption: YAML

    from ads.model.deployment import ModelDeployment

    yaml_string = """
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
          webConcurrency: 10
      runtime:
        kind: runtime
        type: conda
        spec:
          modelUri: <MODEL_URI>
          env:
            WEB_CONCURRENCY: "10"
          deploymentMode: HTTPS_ONLY
    """

    deployment = ModelDeployment.from_yaml(yaml_string)
    deployment.deploy()


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

