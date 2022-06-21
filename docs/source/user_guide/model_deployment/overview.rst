Overview
********

Model deployments are a managed resource within the Oracle Cloud Infrastructure (OCI) Data Science service.  They allow you to deploy machine learning models as web applications (HTTP endpoints). They provide real-time predictions and enables you to quickly productionalize your models.

The ``ads.model.deployment`` module allows you to deploy models using the Data Science service. This module is built on top of the ``oci`` Python SDK. It is designed to simplify data science workflows.

A `model artifact <https://docs.oracle.com/en-us/iaas/data-science/using/models-prepare-artifact.htm>`__ is a ZIP archive of the files necessary to deploy your model. The model artifact contains the `score.py <https://docs.oracle.com/en-us/iaas/data-science/using/model_score_py.htm>`__ file. This file has the Python code that is used to load the model and perform predictions. The model artifact also contains the `runtime.yaml <https://docs.oracle.com/en-us/iaas/data-science/using/model_runtime_yaml.htm>`__ file.  This file is used to define the conda environment used by the model deployment.

ADS supports deploying a model artifact from the Data Science `model catalog <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/modelcatalog/modelcatalog.html>`__, or the URI of a directory that can be in the local block storage or in Object Storage.

You can integrate model deployments with the `OCI Logging service <https://docs.oracle.com/en-us/iaas/data-science/using/log-about.htm#jobs_about__mod-dep-logs>`__.  The system allows you to store access and prediction logs ADS provides APIs to simplify the interaction with the Logging service, see 
`ADS Logging <../logging/logging.html>`__.

The ``ads.model.deployment`` module provides the following classes, which are used to deploy and manage the model.

* ``ModelDeployer``: It creates a new deployment. It is also used to delete, list, and update existing deployments.
* ``ModelDeployment``: Encapsulates the information and actions for an existing deployment.
* ``ModelDeploymentProperties``: Stores the properties used to deploy a model.

