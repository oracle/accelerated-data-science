Quick Start
***********

Model deployments are a managed resource within the Oracle Cloud Infrastructure (OCI) Data Science service.  They allow you to deploy machine learning models as web applications (HTTP endpoints). They provide real-time predictions and enables you to quickly productionalize your models.

The ``ads.model.deployment`` module allows you to deploy models using the Data Science service. This module is built on top of the ``oci`` Python SDK. It is designed to simplify data science workflows.

A `model artifact <https://docs.oracle.com/en-us/iaas/data-science/using/models-prepare-artifact.htm>`__ is a ZIP archive of the files necessary to deploy your model. The model artifact contains the `score.py <https://docs.oracle.com/en-us/iaas/data-science/using/model_score_py.htm>`__ file. This file has the Python code that is used to load the model and perform predictions. The model artifact also contains the `runtime.yaml <https://docs.oracle.com/en-us/iaas/data-science/using/model_runtime_yaml.htm>`__ file.  This file is used to define the conda environment used by the model deployment.

ADS supports deploying a model artifact from the Data Science `model catalog <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/modelcatalog/modelcatalog.html>`__, or the URI of a directory that can be in the local block storage or in Object Storage.

You can integrate model deployments with the `OCI Logging service <https://docs.oracle.com/en-us/iaas/data-science/using/log-about.htm#jobs_about__mod-dep-logs>`__.  The system allows you to store access and prediction logs ADS provides APIs to simplify the interaction with the Logging service, see 
`ADS Logging <../logging/logging.html>`__.

The ``ads.model.deployment`` module provides the ``ModelDeployment`` class, which is used to deploy and manage the model. Checkout the example below to learn how to deploy models.

Example
=======

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

    deployment.deploy() # deploy model
    deployment.with_display_name("Updated name").update() # update deployment with new name
    deployment.predict(data={"line" : "12"}) # predict
    deployment.watch(log_type="access") # stream the access log of deployment
    deployment.delete() # delete the model deployment

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
    deployment.deploy() # deploy model
    deployment.with_display_name("Updated name").update() # update deployment with new name
    deployment.predict(data={"line" : "12"}) # predict
    deployment.watch(log_type="access") # stream the access log of deployment
    deployment.delete() # delete the model deployment


################
Model Deployment
################

.. toctree::
    :maxdepth: 1

    activate
    attributes
    deactivate
    delete
    deploy
    list
    logs
    predict
    properties
    update
