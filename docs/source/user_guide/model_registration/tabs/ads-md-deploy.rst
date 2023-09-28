.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.model.deployment import ModelDeployment, ModelDeploymentInfrastructure, ModelDeploymentContainerRuntime

    # configure model deployment infrastructure
    infrastructure = (
        ModelDeploymentInfrastructure()
        .with_project_id("ocid1.datascienceproject.oc1.<UNIQUE_ID>")
        .with_compartment_id("ocid1.compartment.oc1..<UNIQUE_ID>")
        .with_shape_name("VM.GPU.A10.2")
        .with_bandwidth_mbps(10)
        .with_web_concurrency(10)
        .with_access_log(
            log_group_id="ocid1.loggroup.oc1.<UNIQUE_ID>",
            log_id="ocid1.log.oc1.<UNIQUE_ID>"
        )
        .with_predict_log(
            log_group_id="ocid1.loggroup.oc1.<UNIQUE_ID>",
            log_id="ocid1.log.oc1.<UNIQUE_ID>"
        )
        .with_subnet_id("ocid1.subnet.oc1.<UNIQUE_ID>")
    )

    # ENV_VAR for vllm
    env_var_vllm = {
          "PARAMS": "--model meta-llama/Llama-2-7b-chat-hf",
          "HUGGINGFACE_HUB_CACHE": "/home/datascience/.cache",
          "TOKEN_FILE": "/opt/ds/model/deployed_model/token",
          "STORAGE_SIZE_IN_GB": "950",
          "WEB_CONCURRENCY": 1,
        }

    # ENV_VAR for TGI
    env_var_tgi = {
      "TOKEN": "/opt/ds/model/deployed_model/token",
      "PARAMS": "--model-id meta-llama/Llama-2-7b-chat-hf --max-batch-prefill-tokens 1024",
    }

    # configure model deployment runtime
    container_runtime = (
        ModelDeploymentContainerRuntime()
        .with_image("iad.ocir.io/<namespace>/<image>:<tag>")
        .with_server_port(5001)
        .with_health_check_port(5001)
        .with_env(env_var_vllm) # for TGI, replace with env_var_tgi.
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_uri("ocid1.datasciencemodel.oc1.<UNIQUE_ID>")
        .with_auth({"auth_key":"auth_value"})
        .with_region("us-ashburn-1")
        .with_overwrite_existing_artifact(True)
        .with_remove_existing_artifact(True)
        .with_timeout(100)
    )

    # configure model deployment
    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags({"key1":"value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(container_runtime)
    )

  .. code-tab:: yaml
    :caption: TGI-YAML

    kind: deployment
    spec:
      displayName: LLama2-7b model deployment - tgi
      infrastructure:
        kind: infrastructure
        type: datascienceModelDeployment
        spec:
          compartmentId: ocid1.compartment.oc1..<UNIQUE_ID>
          projectId: ocid1.datascienceproject.oc1.<UNIQUE_ID>
          accessLog:
            logGroupId: ocid1.loggroup.oc1.<UNIQUE_ID>
            logId: ocid1.log.oc1.<UNIQUE_ID>
          predictLog:
            logGroupId: ocid1.loggroup.oc1.<UNIQUE_ID>
            logId: ocid1.log.oc1.<UNIQUE_ID>
          shapeName: VM.GPU.A10.2
          replica: 1
          bandWidthMbps: 10
          webConcurrency: 10
          subnetId: ocid1.subnet.oc1.<UNIQUE_ID>
      runtime:
        kind: runtime
        type: container
        spec:
          modelUri: ocid1.datasciencemodel.oc1.<UNIQUE_ID>
          image: <UNIQUE_ID>
          serverPort: 5001
          healthCheckPort: 5001
          env:
            TOKEN: "/opt/ds/model/deployed_model/token"
            PARAMS: "--model-id meta-llama/Llama-2-7b-chat-hf --max-batch-prefill-tokens 1024"
          region: us-ashburn-1
          overwriteExistingArtifact: True
          removeExistingArtifact: True
          timeout: 100
          deploymentMode: HTTPS_ONLY

  .. code-tab:: yaml
    :caption: vllm-YAML

    kind: deployment
    spec:
      displayName: LLama2-7b model deployment - vllm
      infrastructure:
        kind: infrastructure
        type: datascienceModelDeployment
        spec:
          compartmentId: ocid1.compartment.oc1..<UNIQUE_ID>
          projectId: ocid1.datascienceproject.oc1.<UNIQUE_ID>
          accessLog:
            logGroupId: ocid1.loggroup.oc1.<UNIQUE_ID>
            logId: ocid1.log.oc1.<UNIQUE_ID>
          predictLog:
            logGroupId: ocid1.loggroup.oc1.<UNIQUE_ID>
            logId: ocid1.log.oc1.<UNIQUE_ID>
          shapeName: VM.GPU.A10.2
          replica: 1
          bandWidthMbps: 10
          webConcurrency: 10
          subnetId: ocid1.subnet.oc1.<UNIQUE_ID>
      runtime:
        kind: runtime
        type: container
        spec:
          modelUri: ocid1.datasciencemodel.oc1.<UNIQUE_ID>
          image: <UNIQUE_ID>
          serverPort: 5001
          healthCheckPort: 5001
          env:
            PARAMS: "--model meta-llama/Llama-2-7b-chat-hf"
            HUGGINGFACE_HUB_CACHE: "/home/datascience/.cache"
            TOKEN_FILE: /opt/ds/model/deployed_model/token
            STORAGE_SIZE_IN_GB: "950"
            WEB_CONCURRENCY:  1
          region: us-ashburn-1
          overwriteExistingArtifact: True
          removeExistingArtifact: True
          timeout: 100
          deploymentMode: HTTPS_ONLY
