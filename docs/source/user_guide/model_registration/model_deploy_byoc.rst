Deploy Model with Container Runtime
***********************************

The ADS ``GenericModel`` and ``ModelDeployment`` classes allow you to deploy model using container images.

To deploy model on container runtime, you need to first build a docker container image. See `Bring Your Own Container <https://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm#construct-container>`_ for the end-to-end example. Once you have the image, push it to `OCI container registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_. See `Creating a Repository <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ and `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ for more details.

Deploy Using GenericModel Class
===============================

When the container runtime is ready, you can call ``deploy`` function to deploy the model and generate the endpoint. You must specify the container ``deployment_image``. You can optionally specify the `entrypoint` and `cmd` for running the container (See `Understand how CMD and ENTRYPOINT interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_). For more details regarding the parameters allowed for container runtime, see `BYOC Required Interfaces <https://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm#model-dep-byoc-interfaces>`_.

Below is an example of deploying Sklearn model on container runtime using ``SklearnModel`` class:

.. code-block:: python3

    import os
    import pandas as pd
    from joblib import load
    from ads.model import SklearnModel

    # Load data
    data = pd.read_json(<path_to_data>)
    data_test = data.transpose()
    X = data_test.drop(data_test.loc[:, "Line":"# Letter"].columns, axis=1)
    X_test = X.iloc[int("12"), :].values.reshape(1, -1)

    # Load model
    clf_lda = load(<path_to_model>)

    # Instantiate ads.model.SklearnModel
    sklearn_model = SklearnModel(estimator=clf_lda, artifact_dir=<path_to_artifact_directory>)

    # Prepare related artifacts
    sklearn_model.prepare(
        model_file_name=<model_file_name>,
        ignore_conda_error=True, # make sure to set ignore_conda_error=True for container runtime
    )

    # Verify model locally
    sklearn_model.verify(X_test)

    # Register Sklearn model
    sklearn_model.save()

    # Deploy Sklearn model on container runtime
    sklearn_model.deploy(
        display_name="Sklearn Model BYOC",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_image="iad.ocir.io/<namespace>/<image>:<tag>",
        entrypoint=["python", "/opt/ds/model/deployed_model/api.py"],
        server_port=5000,
        health_check_port=5000,
        environment_variables={"test_key": "test_value"},
    )

    # Get endpoint of deployed model
    model_deployment_url = sklearn_model.model_deployment.url

    # Generate prediction by invoking the deployed endpoint
    sklearn_model.predict(data={"line": "12"})


Deploy Using ModelDeployment Class
==================================

To deploy a model deployment, you can define a ``ModelDeployment`` object and call the ``.deploy()`` of it. You could either use API or YAML to define the ``ModelDeployment`` object.

When configuring the ``ModelDeploymentContainerRuntime`` object, you must specify the container `image`. You can optionally specify the `entrypoint` and `cmd` for running the container (See `Understand how CMD and ENTRYPOINT interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_). For more details regarding the parameters allowed for container runtime, see `BYOC Required Interfaces <https://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm#model-dep-byoc-interfaces>`_.

Below is an example of deploying model on container runtime using ``ModelDeployment`` class: 

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.model.deployment import ModelDeployment, ModelDeploymentInfrastructure, ModelDeploymentContainerRuntime

    # configure model deployment infrastructure
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

    # configure model deployment runtime
    container_runtime = (
        ModelDeploymentContainerRuntime()
        .with_image("iad.ocir.io/<namespace>/<image>:<tag>")
        .with_image_digest("<IMAGE_DIGEST>")
        .with_entrypoint(["python","/opt/ds/model/deployed_model/api.py"])
        .with_server_port(5000)
        .with_health_check_port(5000)
        .with_env({"key":"value"})
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_uri("<MODEL_URI>")
    )

    # configure model deployment
    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags(**{"key1":"value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(container_runtime)
    )

    # Deploy model on container runtime
    deployment.deploy()

    # Generate prediction by invoking the deployed endpoint
    deployment.predict(data=<data>)

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
          image: iad.ocir.io/<namespace>/<image>:<tag>
          imageDigest: <IMAGE_DIGEST>
          entrypoint: ["python","/opt/ds/model/deployed_model/api.py"]
          serverPort: 5000
          healthCheckPort: 5000
          env:
            WEB_CONCURRENCY: "10"
          deploymentMode: HTTPS_ONLY
    """

    # Initialize ads.ModelDeployment
    deployment = ModelDeployment.from_yaml(yaml_string)
    
    # Deploy model on container runtime
    deployment.deploy()

    # Generate prediction by invoking the deployed endpoint
    deployment.predict(data=<data>)


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


**ADS ModelDeploymentInfrastructure YAML Schema**

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

**ADS ModelDeploymentContainerRuntime YAML Schema**

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
