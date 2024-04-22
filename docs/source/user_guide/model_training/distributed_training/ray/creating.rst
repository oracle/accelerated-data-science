Creating Ray Workloads
-----------------------------

.. include:: ../_prerequisite.rst


**Write your training code:**

Here is a sample of Python code (performing Grid Search) leveraging Scikit-Learn
and Ray to distribute the workload:

.. code-block:: python

   import ray
   from sklearn.datasets import make_classification
   from sklearn.svm import SVC
   from sklearn.model_selection import GridSearchCV

   import os
   import argparse

   ray.init(address='auto', ignore_reinit_error=True)

   @ray.remote
   def gridsearch(args):
       grid = GridSearchCV(
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
       return grid.best_params_

   default_n_samples = int(os.getenv("DEFAULT_N_SAMPLES", "1000"))

   parser = argparse.ArgumentParser()
   parser.add_argument("--n_samples", default=default_n_samples, type=int, help="size of dataset")
   parser.add_argument("--cv", default=3, type=int, help="number of cross validations")
   args, unknownargs = parser.parse_known_args()

   # Using environment variable to fetch the SCHEDULER_IP is important

   X, y = make_classification(n_samples=args.n_samples, random_state=42)

   refs = []
   for i in range(0, 5):
       refs.append(gridsearch.remote(args))

   best_params = []
   for ref in refs:
       best_params.append(ray.get(ref))

   print(best_params)

**Initialize a distributed-training folder:**

At this point you have created a training file (or files) - ``gridsearch.py`` from the above
example. Now, run the command below.

.. code-block:: bash

  ads opctl distributed-training init --framework ray --version v1


This will download the ``ray`` framework and place it inside ``'oci_dist_training_artifacts'`` folder.

**Note**: Whenever you change the code, you have to build, tag and push the image to repo. This is automatically done in ```ads opctl run``` cli command.

**Containerize your code and build container:**

The required python dependencies are provided inside the conda environment file `oci_dist_training_artifacts/ray/v1/environments.yaml`.  If your code requires additional dependency, update this file.

Also, while updating `environments.yaml` do not remove the existing libraries. You can append to the list.

Update the TAG and the IMAGE_NAME as per your needs -

.. code-block:: bash

  export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
  export TAG=latest
  export MOUNT_FOLDER_PATH=.

Build the container image.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/ray/v1/Dockerfile \

The code is assumed to be in the current working directory. To override the source code directory, use the ``-s`` flag and specify the code dir. This folder should be within the current working directory.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/ray/v1/Dockerfile \
      -s $MOUNT_FOLDER_PATH

If you are behind proxy, ads opctl will automatically use your proxy settings (defined via ``no_proxy``, ``http_proxy`` and ``https_proxy``).

**Define your workload yaml:**

The ``yaml`` file is a declarative way to express the workload.
In this example, we bring up 1 worker node and 1 chief-worker node.
The training code to run is ``train.py``.
All your training code is assumed to be present inside ``/code`` directory within the container.
Additionally, you can also put any data files inside the same directory
(and pass on the location ex ``/code/data/**`` as an argument to your training script using runtime->spec->args).
This particular configuration will run with 2 nodes. 

.. code-block:: yaml

   # Example train.yaml for defining ray cluster
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
       kind: ray
       apiVersion: v1.0
       spec:
         image: "@image"
         workDir: "oci://my-bucket@my-namespace/rayexample/001"
         name: GridSearch Ray
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

**Note**: make sure that the ``workDir`` points to your object storage
bucket at OCI.

For ``flex shapes`` use following in the ``train.yaml`` file

.. code:: yaml

   shapeConfigDetails:
       memoryInGBs: 22
       ocpus: 2
   shapeName: VM.Standard.E3.Flex


**Use ads opctl to create the cluster infrastructure and run the workload:**

Do a dry run to inspect how the yaml translates to Job and Job Runs

.. code-block:: bash

  ads opctl run -f train.yaml --dry-run


.. include:: ../_test_and_submit.rst

**Monitoring the workload logs**

To view the logs from a job run, you could run -

.. code-block:: bash

  ads opctl watch oci.xxxx.<job_run_ocid>

You could stream the logs from any of the job run ocid using ``ads opctl watch`` command. You could run this command from multiple terminal to watch all of the job runs. Typically, watching ``mainJobRunId`` should yield most informative log.