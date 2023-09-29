====================
Large Language Model
====================

Oracle ADS (Accelerated Data Science) opens the gateway to harnessing the full potential of the Large Language models
within Oracle Cloud Infrastructure (OCI). `Meta <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_'s
latest offering, `Llama 2 <https://ai.meta.com/llama/>`_, introduces a collection of pre-trained and
fine-tuned generative text models, ranging from 7 to 70 billion parameters. These models represent a significant leap
forward, being trained on 40% more tokens and boasting an extended context length of 4,000 tokens.

Throughout this documentation, we showcase two essential inference frameworks:

- `Text Generation Inference (TGI) <https://github.com/huggingface/text-generation-inference>`_. A purpose-built solution for deploying and serving LLMs from Hugging Face, which we extend to meet the interface requirements of model deployment resources.

- `vLLM <https://vllm.readthedocs.io/>`_. An open-source, high-throughput, and memory-efficient inference and serving engine for LLMs from UC Berkeley.


While our primary focus is on the Llama 2 family, the methodology presented here can be applied to other LLMs as well.


**Sample Code**

For your convenience, we provide sample code and a complete walkthrough, available in the `Oracle
GitHub samples repository <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/model-deployment/containers/llama2>`_.

**Prerequisites**

Using the Llama 2 model requires user agreement acceptance on `Meta's website <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_. Downloading the model
from `Hugging Face <https://huggingface.co/meta-llama>`_ necessitates an account and agreement to the service terms. Ensure that the model's license permits
usage for your intended purposes.

**Recommended Hardware**

We recommend specific OCI shapes based on Nvidia A10 GPUs for deploying models. These shapes
cater to both the 7-billion and 13-billion parameter models, with the latter utilizing quantization techniques to
optimize GPU memory usage. OCI offers `a variety of GPU options <https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/Compute/References/computeshapes.htm>`_ to suit your needs.

**Deployment Approaches**

You can use the following methods to deploy an LLM with OCI Data Science:

- Online Method. This approach involves downloading the LLM directly from the hosting repository into the `Data Science Model Deployment <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm>`_. It minimizes data copying, making it suitable for large models. However, it lacks governance and may not be ideal for production environments or fine-tuning scenarios.

- Offline Method. In this method, you download the LLM model from the host repository and save it in the `Data Science Model Catalog <https://docs.oracle.com/en-us/iaas/data-science/using/models-about.htm>`_. Deployment then occurs directly from the catalog, allowing for better control and governance of the model.

**Inference Container**

We explore two inference options: Hugging Face's Text Generation Inference (TGI) and vLLM from UC Berkeley. These
containers are crucial for effective model deployment and are optimized to align with OCI Data Science model deployment requirements.
You can find both the TGI and vLLM Docker files in `our samples repository <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/model-deployment/containers/llama2>`_.

**Creating the Model Deployment**

The final step involves deploying the model and the inference container by creating a model deployment. Once deployed,
the model is accessible via a predict URL, allowing HTTP-based model invocation.

**Testing the Model**

To validate your deployed model, a Gradio Chat app can be configured to use the predict URL. This app provides
parameters such as ``max_tokens``, ``temperature``, and ``top_p`` for fine-tuning model responses. Check our `blog <https://blogs.oracle.com/ai-and-datascience/post/llama2-oci-data-science-cloud-platform>`_ to
learn more about this.


Train Model
-----------

Check `Training Large Language Model <../model_training/training_llm.rst>`_ to see how to train your large language model
by Oracle Cloud Infrastructure (OCI) `Data Science Jobs (Jobs) <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`_.


Register Model
--------------

Once you've trained your LLM, we guide you through the process of registering it within OCI, enabling seamless access and management.

Zip all items of the folder using zip/tar utility, preferrably using below command to avoid creating another hierarchy of folder structure inside zipped file.

.. code-block:: bash

    zip my_large_model.zip * -0

Upload the zipped artifact created in an object storage bucket in your tenancy. Tools like `rclone <https://rclone.org/>`_,
can help speed this upload. Using rclone with OCI can be referred from `here <https://docs.oracle.com/en/solutions/move-data-to-cloud-storage-using-rclone/configure-rclone-object-storage.html#GUID-8471A9B3-F812-4358-945E-8F7EEF115241>`_.

Example of using ``oci-cli``:

.. code-block:: bash

    oci os object put -ns <namespace> -bn <bucket> --name <prefix>/my_large_model.zip --file my_large_model.zip

Next step is to create a model catalog item. Use :py:class:`~ads.model.DataScienceModel` to register the large model to Model Catalog.

.. versionadd:: 2.8.10

.. code-block:: python

    import ads
    from ads.model import DataScienceModel

    ads.set_auth("resource_principal")

    MODEL_DISPLAY_NAME = "My Large Model"
    ARTIFACT_PATH = "oci://<bucket>@<namespace>/<prefix>/my_large_model.zip"

    model = (DataScienceModel()
            .with_display_name(MODEL_DISPLAY_NAME)
            .with_artifact(ARTIFACT_PATH)
            .create(
                remove_existing_artifact=False
            ))
    model_id = model.id

Deploy Model
------------

The final step involves deploying your registered LLM for real-world applications. We walk you through deploying it in a
`custom containers (Bring Your Own Container) <http://docs.oracle.com/en-us/iaas/data-science/using/mod-dep-byoc.htm>`_ within the OCI Data
Science Service, leveraging advanced technologies for optimal performance.

You can define the model deployment with `ADS Python APIs <../model_registration/model_deploy_byoc.rst>`_ or YAML. In the
examples below, you will need to change with the OCIDs of the resources required for the deployment, like ``project ID``,
``compartment ID`` etc. All of the configurations with ``<UNIQUE_ID>`` should be replaces with your corresponding ID from
your tenancy, the resources we created in the previous steps.


Online Deployment
^^^^^^^^^^^^^^^^^

**Prerequisites**

Check on `GitHub Sample repository <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/model-deployment/containers/llama2#model-deployment-steps>`_ to see how to complete the Prerequisites before actual deployment.

- Zips your Hugging Face user access token and registers it into Model Catalog by following the instruction on ``Register Model`` in this page.
- Creates logging in the `OCI Logging Service <https://cloud.oracle.com/logging/log-groups>`_ for the model deployment (if you have to already created, you can skip this step).
- Creates a subnet in `Virtual Cloud Network <https://cloud.oracle.com/networking/vcns>`_  for the model deployment.
- Executes container build and push process to `Oracle Cloud Container Registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_.
- You can now use the Bring Your Own Container Deployment in OCI Data Science to the deploy the Llama2 model.

.. include:: ../model_registration/tabs/env-var-online.rst

.. include:: ../model_registration/tabs/ads-md-deploy-online.rst

Offline Deployment
^^^^^^^^^^^^^^^^^^

Check on `GitHub Sample repository <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/model-deployment/containers/llama2-offline#model-deployment-steps>`_ to see how to complete the Prerequisites before actual deployment.

- Registers the zipped artifact into Model Catalog by following the instruction on ``Register Model`` in this page.
- Creates logging in the `OCI Logging Service <https://cloud.oracle.com/logging/log-groups>`_ for the model deployment (if you have to already created, you can skip this step).
- Executes container build and push process to `Oracle Cloud Container Registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_.
- You can now use the Bring Your Own Container Deployment in OCI Data Science to the deploy the Llama2 model.

.. include:: ../model_registration/tabs/env-var-offline.rst

.. include:: ../model_registration/tabs/ads-md-deploy-offline.rst

You can deploy the model through API call or ADS CLI.

Make sure that you've also created and setup your `API Auth Token <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrygettingauthtoken.htm>`_ to execute the commands below.

.. include:: ../model_registration/tabs/run_md.rst


Inference Model
---------------

Once the model is deployed and shown as Active you can execute inference against it. You can run inference against
the deployed model with oci-cli from your OCI Data Science Notebook or you local environment.

.. include:: ../model_registration/tabs/run_predict.rst
