Oracle Cloud Infrastructure (OCI) `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_
enables you to define and run repeatable machine learning tasks on a fully managed infrastructure.
You can create a compute resource on demand and run applications that perform tasks such as
data preparation, model training, hyperparameter tuning, and batch inference.

Running workloads with Data Science Jobs involves two types resources: **Job** and **Job Run**.

A **Job** is a template that describes the training task.
It contains configurations about the *infrastructure*, such as
`Compute Shape <https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm>`_,
`Block Storage <https://docs.oracle.com/en-us/iaas/Content/Block/Concepts/overview.htm>`_,
`Logging <https://docs.oracle.com/en-us/iaas/Content/Logging/Concepts/loggingoverview.htm>`_,
and information about the *runtime*,
such as the source code of your workload, environment variables, and CLI arguments.

A **Job Run** is an instantiation of a job.
In each job run, you can override some of the job configurations, such as environment variables and CLI arguments.
You can use the same job as a template and launch multiple simultaneous job runs to parallelize a large task.
You can also sequence jobs and keep the state by writing state information to
`Object Storage <https://docs.oracle.com/en-us/iaas/Content/Object/Concepts/objectstorageoverview.htm>`_

For example, you may want to experiment with how different model classes perform on the same training data
by using the `ADSTuner <https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/ads_tuner.html>`_
to perform hyperparameter tuning on each model class.
You could do this in parallel by having a different job run for each class of models.
For a given job run, you could pass an environment variable that identifies the model class that you want to use.
Each model can write its results to the Logging service or Object Storage.
Then you can run a final sequential job that uses the best model class, and trains the final model on the entire dataset.

The following sections provides details on running workloads with OCI Data Science Jobs using ADS Jobs APIs.
You can use similar APIs to `Run a OCI DataFlow Application <run_data_flow.html>`_.
