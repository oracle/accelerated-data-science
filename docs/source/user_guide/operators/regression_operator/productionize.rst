==============
Productionize
==============

This page covers the production-facing parts of the current regression operator implementation: local container runs, OCI Data Science Job backends, and optional save-and-deploy behavior.

Generate Runtime Configs
------------------------

Start by generating the operator configs:

.. code-block:: bash

    ads operator init -t regression --overwrite --output ~/regression/

You will always get:

* ``regression.yaml``
* ``regression_operator_local_python_backend.yaml``
* ``regression_operator_local_container_backend.yaml``

If your ADS CLI defaults are configured for Jobs, you also get:

* ``regression_job_container_backend.yaml``
* ``regression_job_python_backend.yaml``

Local Container Backend
-----------------------

Build the operator image:

.. code-block:: bash

    ads operator build-image -t regression

The generated local container backend file looks like this:

.. code-block:: yaml

    kind: operator.local
    spec:
      env:
      - name: operator
        value: regression:v1
      image: regression:v1
      volume:
      - /Users/<user>/.oci:/root/.oci
    type: container
    version: v1

If your training and output paths are local file paths, mount those directories into the container and update ``regression.yaml`` to use the in-container paths.

Run the container backend:

.. code-block:: bash

    ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_operator_local_container_backend.yaml

OCI Data Science Jobs
---------------------

Container Runtime
~~~~~~~~~~~~~~~~~

Build and publish the image:

.. code-block:: bash

    ads operator build-image -t regression
    ads operator publish-image -t regression --registry <iad.ocir.io/tenancy/>

Then run with the generated backend config:

.. code-block:: bash

    ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_job_container_backend.yaml

Python/Conda Runtime
~~~~~~~~~~~~~~~~~~~~

Build and publish the conda pack:

.. code-block:: bash

    ads operator build-conda -t regression
    ads operator publish-conda -t regression

Then run:

.. code-block:: bash

    ads operator run -f ~/regression/regression.yaml -b ~/regression/regression_job_python_backend.yaml

Using Object Storage Paths
--------------------------

When the operator runs as a job, it cannot read your local machine paths. Update the data and output locations to ``oci://`` URIs.

Example:

.. code-block:: yaml

    kind: operator
    type: regression
    version: v1
    spec:
      training_data:
        url: oci://bucket@namespace/regression/input/train.csv
      test_data:
        url: oci://bucket@namespace/regression/input/test.csv
      output_directory:
        url: oci://bucket@namespace/regression/output/
      target_column: target
      model: random_forest

After submission, monitor logs with:

.. code-block:: bash

    ads opctl watch <OCID>

Save to Model Catalog and Deploy
--------------------------------

The regression operator can optionally save the trained model artifact to OCI Model Catalog and create a Model Deployment.

Example:

.. code-block:: yaml

    save_and_deploy_to_md:
      model_catalog_display_name: regression-rf-model
      model_deployment:
        display_name: regression-rf-md
        initial_shape: VM.Standard.E4.Flex
        description: Regression model deployment

Notes from the current implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``project_id`` defaults from the ``PROJECT_OCID`` environment variable if not provided.
* ``compartment_id`` defaults from ``NB_SESSION_COMPARTMENT_OCID`` if not provided.
* ``model.pkl`` is always written first and then used for model packaging.
* The deployment artifact bundles ``models.pickle`` plus ``score.py`` for inference.
* When ``save_and_deploy_to_md`` is present, the run writes ``model_registration_info.json``.
* The deployment manager also writes ``deployment_info.json``.

Autoscaling and Logging
-----------------------

The deployment block also supports autoscaling and OCI Logging fields:

.. code-block:: yaml

    save_and_deploy_to_md:
      model_catalog_display_name: regression-rf-model
      model_deployment:
        display_name: regression-rf-md
        initial_shape: VM.Standard.E4.Flex
        log_group: ocid1.loggroup.oc1..example
        log_id: ocid1.log.oc1..example
        auto_scaling:
          minimum_instance: 1
          maximum_instance: 2
          scale_in_threshold: 10
          scale_out_threshold: 80
          scaling_metric: CPU_UTILIZATION
          cool_down_in_seconds: 600

Inference Payload Shape
-----------------------

The packaged deployment ``score.py`` expects a payload with a top-level ``data`` key.

Example:

.. code-block:: json

    {
      "data": [
        {"x1": 1.2, "x2": 3.4},
        {"x1": 5.6, "x2": 7.8}
      ]
    }

If the payload also contains the training target column, the deployment scorer drops that column before prediction.
