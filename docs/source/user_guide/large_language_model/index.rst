.. _large_language_model:

####################
Large Language Model
####################

Oracle Cloud Infrastructure (OCI) provides fully managed infrastructure to work with Large Language Model (LLM). 

Train and Deploy LLM
********************
You can train LLM at scale with multi-node and multi-GPU using `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_, and deploy it with `Data Science Model Deployment (Model Deployments) <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm>`_. The following blog posts show examples training and deploying Llama2 models:

* `Multi-GPU multinode fine-tuning Llama2 on OCI Data Science <https://blogs.oracle.com/ai-and-datascience/post/multi-gpu-multi-node-finetuning-llama2-oci>`_
* `Deploy Llama 2 in OCI Data Science <https://blogs.oracle.com/ai-and-datascience/post/llama2-oci-data-science-cloud-platform>`_
* `Quantize and deploy Llama 2 70B on cost-effective NVIDIA A10 Tensor Core GPUs in OCI Data Science <https://blogs.oracle.com/ai-and-datascience/post/quantize-deploy-llama2-70b-costeffective-a10s-oci>`_


Integration with LangChain
**************************
ADS is designed to work with LangChain, enabling developers to incorporate various LangChain components and models deployed on OCI seamlessly into their applications. Additionally, ADS can package LangChain applications and deploy it as a REST API endpoint using OCI Data Science Model Deployment.


.. admonition:: Installation
  :class: note

  Install ADS and other dependencies for LLM integrations.

  .. code-block:: bash

    $ python3 -m pip install "oracle-ads[llm]"



.. toctree::
    :hidden:
    :maxdepth: 2

    training_llm
    deploy_langchain_application
    retrieval