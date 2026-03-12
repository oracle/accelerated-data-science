============================================
Upload a Model Group artifact (homogeneous)
============================================

A Model Group is a logical construct used to encapsulate several machine learning models into a single, version-controlled unit.
With a model Group, you can group deployments, share resources, and perform live updates while maintaining immutability and reproducibility.
In ADS, a Model Group is represented by :py:class:`ads.model.datascience_model_group.DataScienceModelGroup`.

Model Group types
=================

Model Groups can be created in different forms depending on your deployment pattern:

* **Homogeneous**: A group of member models that share the same runtime and can be deployed together.
  For homogeneous model groups, ADS supports attaching a **deployment runtime artifact** (``score.py`` + ``runtime.yaml``)
  to the group.
* **Stacked**: A group with a designated **base model** (``base_model_id``) plus additional member models.
  This is commonly used for stacked deployments.

Common operations
=================

The :py:class:`~ads.model.datascience_model_group.DataScienceModelGroup` API supports standard lifecycle operations:

* ``create()`` / ``update()`` / ``delete()``
* ``activate()`` / ``deactivate()``
* ``from_id(<model_group_ocid>)``
* ``list(...)``

Member models
=============

Member models are provided via ``with_member_models(...)`` as a list of dictionaries:

* ``model_id``: The model OCID.
* ``inference_key``: A short name used to identify the model within the group.


Deployment types and runtime types
=================================

In OCI Data Science, deployments can be created for:

* A **single model** (``ModelDeploymentContainerRuntime.with_model_uri(<model_ocid>)``), or
* A **model group** (``ModelDeploymentContainerRuntime.with_model_group_id(<model_group_ocid>)``).

ADS supports different runtime options for model deployments. Two common patterns are:

* **Conda-based runtime** using a standard model artifact (``score.py`` + ``runtime.yaml``).
* **Container runtime (BYOC)** where you specify the container image and runtime configuration.

The examples below focus on container runtime deployment using a model group.


Artifact requirements
====================

The artifact must be either:

* A **directory** containing the required files at the **top level**, or
* A **.zip** file containing the same structure.

At minimum, the following files must exist:

* ``score.py``
* ``runtime.yaml``

Example usage
=============

Homogeneous model group (with runtime artifact)
----------------------------------------------

.. code-block:: python

    import os

    from ads.model.datascience_model_group import DataScienceModelGroup
    from ads.model.model_metadata import ModelCustomMetadata


    # Path to a model deployment runtime artifact directory.
    # Example layout:
    #   ./group_runtime_artifact/
    #     score.py
    #     runtime.yaml
    artifact_dir = "./group_runtime_artifact"

    custom_metadata = ModelCustomMetadata()
    custom_metadata.add(
        key="test_key",
        value="test_value",
        description="test_description",
        category="other"
    )

    model_group = (
        DataScienceModelGroup()
        .with_compartment_id(os.environ.get("NB_SESSION_COMPARTMENT_OCID"))
        .with_project_id(os.environ.get("PROJECT_OCID"))
        .with_display_name("test-model-group")
        .with_description("Homogeneous model group with runtime artifact")
        .with_custom_metadata_list(custom_metadata)
        .with_member_models(
            [
                {"inference_key": "model_a", "model_id": "<model_ocid_a>"},
                {"inference_key": "model_b", "model_id": "<model_ocid_b>"},
            ]
        )
        .with_artifact(artifact_dir)
    )

    # For homogeneous model groups, `create()` uploads the artifact after the group is created.
    model_group.create()


Stacked model group (no group artifact)
--------------------------------------

For stacked model groups, you provide a ``base_model_id``.
The model group artifact upload is **only** applicable for homogeneous model groups.

.. code-block:: python

    import os

    from ads.model.datascience_model_group import DataScienceModelGroup
    from ads.model.model_metadata import ModelCustomMetadata


    custom_metadata = ModelCustomMetadata()
    custom_metadata.add(
        key="test_key",
        value="test_value",
        description="test_description",
        category="other",
    )

    base_model_id = "<base_model_ocid>"

    stacked_group = (
        DataScienceModelGroup()
        .with_compartment_id(os.environ.get("NB_SESSION_COMPARTMENT_OCID"))
        .with_project_id(os.environ.get("PROJECT_OCID"))
        .with_display_name("test-stacked-model-group")
        .with_description("Stacked model group")
        .with_custom_metadata_list(custom_metadata)
        .with_base_model_id(base_model_id)
        .with_member_models(
            [
                {"inference_key": "base", "model_id": base_model_id},
                {"inference_key": "adapter_1", "model_id": "<adapter_model_ocid_1>"},
            ]
        )
    )

    stacked_group.create()


Deploy a Model Group using container runtime
===========================================

The following example shows how to create a model group and deploy it using a **custom container runtime**.

.. code-block:: python

    from ads.model.datascience_model_group import DataScienceModelGroup
    from ads.model.model_metadata import ModelCustomMetadata
    from ads.model.deployment import (
        ModelDeployment,
        ModelDeploymentInfrastructure,
        ModelDeploymentContainerRuntime,
    )


    custom_metadata = ModelCustomMetadata()
    custom_metadata.add(
        key="test_key",
        value="test_value",
        description="test_description",
        category="other",
    )

    model_group = (
        DataScienceModelGroup()
        .with_display_name("test_create_model_group")
        .with_description("test create model group description")
        .with_freeform_tags(**{"test_key": "test_value"})
        .with_custom_metadata_list(custom_metadata)
        .with_member_models(
            [
                {
                    "inference_key": "meta-llama/Llama-2-7b-hf",
                    "model_id": "ocid1.datasciencemodel.oc1.<region>.<unique_id>",
                },
                {
                    "inference_key": "gemma-2b-gov-ext",
                    "model_id": "ocid1.datasciencemodel.oc1.<region>.<unique_id>",
                },
            ]
        )
    )
    model_group.create()


    # Configure model deployment infrastructure
    infrastructure = (
        ModelDeploymentInfrastructure()
        .with_project_id("<PROJECT_OCID>")
        .with_compartment_id("<COMPARTMENT_OCID>")
        .with_shape_name("VM.Standard.E4.Flex")
        .with_shape_config_details(ocpus=1, memory_in_gbs=16)
        .with_replica(1)
        .with_bandwidth_mbps(10)
        .with_web_concurrency(10)
        .with_access_log(
            log_group_id="<ACCESS_LOG_GROUP_OCID>",
            log_id="<ACCESS_LOG_OCID>",
        )
        .with_predict_log(
            log_group_id="<PREDICT_LOG_GROUP_OCID>",
            log_id="<PREDICT_LOG_OCID>",
        )
        .with_subnet_id("<SUBNET_OCID>")
    )


    # Configure model deployment runtime
    container_runtime = (
        ModelDeploymentContainerRuntime()
        .with_image("<region>.ocir.io/<namespace>/<image>:<tag>")
        .with_image_digest("<IMAGE_DIGEST>")
        .with_entrypoint(["python", "/opt/ds/model/deployed_model/api.py"])
        .with_server_port(5000)
        .with_health_check_port(5000)
        .with_env({"key": "value"})
        .with_deployment_mode("HTTPS_ONLY")
        .with_model_group_id(model_group.id)
    )


    # Configure model deployment
    deployment = (
        ModelDeployment()
        .with_display_name("Model Deployment Demo using ADS")
        .with_description("The model deployment description")
        .with_freeform_tags(**{"key1": "value1"})
        .with_infrastructure(infrastructure)
        .with_runtime(container_runtime)
    )

    # Deploy
    deployment.deploy()

    # Invoke endpoint
    deployment.predict(data=<data>)
