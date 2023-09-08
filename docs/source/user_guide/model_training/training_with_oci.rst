#################
Training with OCI
#################

Oracle Cloud Infrastructure (OCI) `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_
enables you to define and run repeatable machine learning tasks on a fully managed infrastructure.
You can have Compute resource on demand and run applications that perform tasks such as
data preparation, model training, hyperparameter tuning, and batch inference.

Here is an example for training RNN on `Word-level Language Modeling <https://github.com/pytorch/examples/tree/main/word_language_model>`_,
using the source code directly from GitHub.

.. include:: ../jobs/tabs/training_job.rst

.. include:: ../jobs/tabs/run_job.rst

The job run will:

* Setup the PyTorch conda environment
* Fetch the source code from GitHub
* Run the training script with the specific arguments
* Save the outputs to OCI object storage

Following are the example outputs of the job run:

.. code-block:: text

    2023-02-27 20:26:36 - Job Run ACCEPTED
    2023-02-27 20:27:05 - Job Run ACCEPTED, Infrastructure provisioning.
    2023-02-27 20:28:27 - Job Run ACCEPTED, Infrastructure provisioned.
    2023-02-27 20:28:53 - Job Run ACCEPTED, Job run bootstrap starting.
    2023-02-27 20:33:05 - Job Run ACCEPTED, Job run bootstrap complete. Artifact execution starting.
    2023-02-27 20:33:08 - Job Run IN_PROGRESS, Job run artifact execution in progress.
    2023-02-27 20:33:31 - | epoch   1 |   200/ 2983 batches | lr 20.00 | ms/batch  8.41 | loss  7.63 | ppl  2064.78
    2023-02-27 20:33:32 - | epoch   1 |   400/ 2983 batches | lr 20.00 | ms/batch  8.23 | loss  6.86 | ppl   949.18
    2023-02-27 20:33:34 - | epoch   1 |   600/ 2983 batches | lr 20.00 | ms/batch  8.21 | loss  6.47 | ppl   643.12
    2023-02-27 20:33:36 - | epoch   1 |   800/ 2983 batches | lr 20.00 | ms/batch  8.22 | loss  6.29 | ppl   537.11
    2023-02-27 20:33:37 - | epoch   1 |  1000/ 2983 batches | lr 20.00 | ms/batch  8.22 | loss  6.14 | ppl   462.61
    2023-02-27 20:33:39 - | epoch   1 |  1200/ 2983 batches | lr 20.00 | ms/batch  8.21 | loss  6.05 | ppl   425.85
    ...
    2023-02-27 20:35:41 - =========================================================================================
    2023-02-27 20:35:41 - | End of training | test loss  4.96 | test ppl   142.94
    2023-02-27 20:35:41 - =========================================================================================
    ...

For more details, see:

* :doc:`../jobs/index`
* :doc:`../jobs/run_git`
