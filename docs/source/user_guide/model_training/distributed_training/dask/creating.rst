Creating Workloads
------------------

.. include:: ../_prerequisite.rst


**Write your training code:**

While running distributed workload, the IP address of the scheduler is known only during the runtime. The IP address is exported as environment variable - ``SCHEDULER_IP`` in all the nodes when the Job Run is in `IN_PROGRESS` state.
Create ``dask.distributed.Client`` object using environment variable to specify the IP address.
Eg. -

.. code-block:: python

  client = Client(f"{os.environ['SCHEDULER_IP']}:{os.environ.get('SCHEDULER_PORT','8786')}")

see :doc:`Writing Dask Code<coding>` for more examples.

For this example, the code to run on the cluster will be:

.. code-block:: python
  :caption: gridsearch.py
  :name: gridsearch.py

  from dask.distributed import Client
  from sklearn.datasets import make_classification
  from sklearn.svm import SVC
  from sklearn.model_selection import GridSearchCV

  import pandas as pd
  import joblib
  import os
  import argparse

  default_n_samples = int(os.getenv("DEFAULT_N_SAMPLES", "1000"))

  parser = argparse.ArgumentParser()
  parser.add_argument("--n_samples", default=default_n_samples, type=int, help="size of dataset")
  parser.add_argument("--cv", default=3, type=int, help="number of cross validations")
  args, unknownargs = parser.parse_known_args()

  # Using environment variable to fetch the SCHEDULER_IP is important.
  client = Client(f"{os.environ['SCHEDULER_IP']}:{os.environ.get('SCHEDULER_PORT','8786')}")

  X, y = make_classification(n_samples=args.n_samples, random_state=42)

  with joblib.parallel_backend("dask"):
      GridSearchCV(
          SVC(gamma="auto", random_state=0, probability=True),
          param_grid={
              "C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              "kernel": ["rbf", "poly", "sigmoid"],
              "shrinking": [True, False],
          },
          return_train_score=False,
          cv=args.cv,
          n_jobs=-1,
      ).fit(X, y)

**Initialize a distributed-training folder:**

At this point you have created a training file (or files) - ``gridsearch.py`` in the above
example. Now running the command below

**Note**: This step requires an internet connection. The `init` command initializes your code directory with dask related artifacts to build

.. code-block:: bash

  ads opctl distributed-training init --framework dask


**Containerize your code and build container:**

Before you can build the image, you must set the following environment variables:

Specify image name and tag

.. code-block:: bash

  export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
  export TAG=latest


Build the container image.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/dask/v1/Dockerfile



The code is assumed to be in the current working directory. To override the source code directory, use the ``-s`` flag and specify the code dir. This folder should be within the current working directory.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/dask/v1/Dockerfile
      -s <code_dir>

If you are behind proxy, ads opctl will automatically use your proxy settings (defined via ``no_proxy``, ``http_proxy`` and ``https_proxy``).



**Define your workload yaml:**

The ``yaml`` file is a declarative way to express the workload. Refer :doc:`YAML schema<../yaml_schema>` for more details.

.. code-block:: yaml
  :caption: train.yaml

  kind: distributed
  apiVersion: v1.0
  spec:
    infrastructure:
      kind: infrastructure
      type: dataScienceJob
      apiVersion: v1.0
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: my_distributed_training
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.Standard2.4
        blockStorageSize: 50
    cluster:
      kind: dask
      apiVersion: v1.0
      spec:
        image: my-region.ocir.io/my-tenancy/dask-cluster-examples:dev
        workDir: "oci://my-bucket@my-namespace/daskexample/001"
        name: GridSearch Dask
        main:
            config:
        worker:
            config:
            replicas: 2
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: "gridsearch.py"
        kwargs: "--cv 5"
        env:
          - name: DEFAULT_N_SAMPLES
            value: 5000



**Use ads opctl to create the cluster infrastructure and run the workload:**

Do a dry run to inspect how the yaml translates to Job and Job Runs. This does not create actual Job or Job Run.

.. code-block:: bash

  ads opctl run -f train.yaml --dry-run

This will give an option similar to this -

.. code-block:: bash

  -----------------------------Entering dryrun mode----------------------------------
  Creating Job with payload:
  kind: job
  spec:
    infrastructure:
      kind: infrastructure
      spec:
        blockStorageSize: 50
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: GridSearch Dask
        jobInfrastructureType: ME_STANDALONE
        jobType: DEFAULT
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxxx.<log_ocid>
        projectId: oci.xxxx.<project_ocid>
        shapeName: VM.Standard2.4
        subnetId: oci.xxxx.<subnet-ocid>
      type: dataScienceJob
    name: GridSearch Dask
    runtime:
      kind: runtime
      spec:
        entrypoint: null
        env:
        - name: OCI__WORK_DIR
          value: oci://my-bucket@my-namespace/daskexample/001
        - name: OCI__EPHEMERAL
          value: None
        - name: OCI__CLUSTER_TYPE
          value: DASK
        - name: OCI__WORKER_COUNT
          value: '2'
        - name: OCI__START_ARGS
          value: ''
        - name: OCI__ENTRY_SCRIPT
          value: gridsearch.py
        - name: OCI__ENTRY_SCRIPT_KWARGS
          value: --cv 5
        - name: DEFAULT_N_SAMPLES
          value: '5000'
        image: my-region.ocir.io/my-tenancy/dask-cluster-examples:dev
      type: container

  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Creating Main Job with following details:
  Name: main
  Environment Variables:
      OCI__MODE:MAIN
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Creating 2 worker jobs with following details:
  Name: worker
  Environment Variables:
      OCI__MODE:WORKER
  -----------------------------Ending dryrun mode----------------------------------

.. include:: ../_test_and_submit.rst

**Monitoring the workload logs**

To view the logs from a job run, you could run -

.. code-block:: bash

  ads opctl watch oci.xxxx.<job_run_ocid>

You could stream the logs from any of the job run ocid using ``ads opctl watch`` command. Your could run this comand from mutliple terminal to watch all of the job runs. Typically, watching ``mainJobRunId`` should yeild most informative log.

To find the IP address of the scheduler dashboard, you could check the configuration file generated by the Main job by running -

.. code-block:: bash

  ads opctl distributed-training show-config -f info.yaml

This will generate an output such as follows -

.. code-block:: yaml

  Main Info:
  OCI__MAIN_IP: <ip address>
  SCHEDULER_IP: <ip address>
  tmpdir: oci://my-bucket@my-namesapce/daskcluster-testing/005/oci.xxxx.<job_ocid>

Dask dashboard is host at : ``http://{SCHEDULER_IP}:8787``
If the IP address is reachable from your workstation network, you can access the dashboard directly from your workstation.
The alternate approach is to use either a Bastion host on the same subnet as the Job Runs and create an ssh tunnel from your workstation.

For more information about the dashboard, checkout https://docs.dask.org/en/stable/diagnostics-distributed.html

.. include:: ../_save_artifacts.rst
.. code-block:: python

  with open(os.path.join(os.environ.get("OCI__SYNC_DIR"),"results.txt"), "w") as rf:
    rf.write(f"Best Params are: {grid.best_params_}, Score is {grid.best_score_}")

**Terminating In-Progress Cluster**

To terminate a running cluster, you could run - 

.. code-block:: bash

  ads opctl distributed-training cancel -f info.yaml
