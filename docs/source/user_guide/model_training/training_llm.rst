Training Large Language Model
*****************************

.. versionadded:: 2.8.8

Oracle Cloud Infrastructure (OCI) `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_
provides fully managed infrastructure to enable training large language model at scale.
This page shows an example of fine-tuning the `Llama 2 <https://ai.meta.com/llama/>`_ model. For model details on the APIs, see :doc:`../jobs/run_pytorch_ddp`.

.. admonition:: Distributed Training with OCI Data Science
  :class: note

  You need to configure your `networking <https://docs.oracle.com/en-us/iaas/Content/Network/Concepts/overview.htm>`_
  and `IAM <https://docs.oracle.com/en-us/iaas/Content/Identity/Concepts/overview.htm>`_ policies.
  We recommend running the training on a private subnet.
  In this example, internet access is needed to download the source code and the pre-trained model.

The `llama-recipes <llama-recipes>`_ repository contains example code to fine-tune llama2 model.
The example `fine-tuning script <https://github.com/facebookresearch/llama-recipes/blob/1aecd00924738239f8d86f342b36bacad180d2b3/examples/finetuning.py>`_ supports both full parameter fine-tuning
and `Parameter-Efficient Fine-Tuning (PEFT) <https://huggingface.co/blog/peft>`_.
With ADS, you can start the training job by taking the source code directly from Github with no code change.

Access the Pre-Trained Model
============================

To fine-tune the model, you will first need to access the pre-trained model.
The pre-trained model can be obtained from `Meta <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_
or `HuggingFace <https://huggingface.co/models?sort=trending&search=meta-llama%2Fllama-2>`_.
In this example, we will use the `access token <https://huggingface.co/docs/hub/security-tokens>`_
to download the pre-trained model from HuggingFace (by setting the ``HUGGING_FACE_HUB_TOKEN`` environment variable).

Fine-Tuning the Model
=====================

You can define the training job with ADS Python APIs or YAML. Here the examples for fine-tuning full parameters of the `7B model <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_ using `FSDP <https://engineering.fb.com/2021/07/15/open-source/fsdp/>`_.

.. include:: ../jobs/tabs/llama2_full.rst

You can create and start the job run API call or ADS CLI.

.. include:: ../jobs/tabs/run_job.rst

The job run will:

* Setup the PyTorch conda environment and install additional dependencies.
* Fetch the source code from GitHub and checkout the specific commit.
* Run the training script with the specific arguments, which includes downloading the model and dataset.
* Save the outputs to OCI object storage once the training finishes.

Note that in the training command, there is no need specify the number of nodes, or the number of GPUs. ADS will automatically configure that base on the ``replica`` and ``shape`` you specified.

The fine-tuning runs on the `samsum <https://huggingface.co/datasets/samsum>`_ dataset by default. You can also `add your custom datasets <https://github.com/facebookresearch/llama-recipes/blob/1aecd00924738239f8d86f342b36bacad180d2b3/docs/Dataset.md>`_.

Once the fine-tuning is finished, the checkpoints will be saved into OCI object storage bucket as specified.
You can `load the FSDP checkpoints for inferencing <https://github.com/facebookresearch/llama-recipes/blob/main/docs/inference.md#loading-back-fsdp-checkpoints>`_.

The same training script also support Parameter-Efficient Fine-Tuning (PEFT). You can change the ``command`` to the following for PEFT with `LoRA <https://huggingface.co/docs/peft/conceptual_guides/lora>`_. Note that for PEFT, the fine-tuned weights are stored in the location specified by ``--output_dir``, while for full parameter fine-tuning, the checkpoints are stored in the location specified by ``--dist_checkpoint_root_folder`` and ``--dist_checkpoint_folder``

.. code-block:: bash

    torchrun llama_finetuning.py --enable_fsdp --use_peft --peft_method lora \
    --pure_bf16 --batch_size_training 1 \
    --model_name meta-llama/Llama-2-7b-hf --output_dir /home/datascience/outputs
