##################################
Deploy LLM Applications and Agents
##################################

Oracle ADS supports the deployment of LLM applications and agents, including LangChain application to OCI data science model deployment.

.. admonition:: IAM Policies
  :class: note

  Ensure that you have configured the necessary `policies for model deployments <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth>`_. 
  For example, the following policy allows the dynamic group to use ``resource_principal`` to create model deployment.

  .. code-block:: shell

      allow dynamic-group <dynamic-group-name> to manage data-science-model-deployments in compartment <compartment-name>

The process of deploying LLM apps and agents involves:

* Prepare your applications as model artifact
* Register the model artifact with OCI Data Science Model Catalog
* Build container image with dependencies, and push the image to OCI Container Registry
* Deploy the model artifact using the container image with OCI Data Science Model Deployment

To get you started, we provide templates for `model artifacts <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/LLM/deployment/model_artifacts>`_ and `container image <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/LLM/deployment/container>`_, so that you can focus on building you applications and agents.

.. figure:: figures/workflow.png
  :width: 800

Prepare Model Artifacts
***********************

You can prepare your model artifact based on the `model artifact template <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/LLM/deployment/model_artifacts>`_

First, create a template folder locally with the `score.py <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/LLM/deployment/model_artifacts/score.py>`_ file. For example, we can call it ``llm_apps_template``.

.. code-block::

  llm_apps_template
  ├── score.py

The ``score.py`` serves as an agent for invoking your application with JSON payload.

Next, you can use ADS to create a generic model and save a copy of the template to a anther folder (e.g. ``my_apps``), which will be uploaded as model artifact.

.. code-block:: python

  from ads.model.generic_model import GenericModel

  llm_app = GenericModel.from_model_artifact(
      uri="llm_apps_template", # Contains the model artifact templates
      artifact_dir="my_apps",  # Location for the new model artifacts
      model_input_serializer="cloudpickle"
  )
  llm_app.reload_runtime_info()

Then, you can add your own applications to the my_apps folder. Here are some requirements:
* Each application should be a Python module.
* Each module should have an ``invoke()`` function as the entrypoint.
* The ``invoke()`` function should take a dictionary and return another dictionary.

For example, following is an example LangChain application to translate English into French using a prompt template and output parser:

.. code-block:: python
  
  import os
  import ads
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from ads.llm import ChatOCIModelDeploymentVLLM


  ads.set_auth(auth="resource_principal")


  llm = ChatOCIModelDeploymentVLLM(
      model="odsc-llm",
      # LLM_ENDPOINT environment variable should be set to a model deployment endpoint.
      endpoint=os.environ["LLM_ENDPOINT"],
      # Optionally you can specify additional keyword arguments for the model, e.g. temperature.
      temperature=0.1,
  )

  prompt = ChatPromptTemplate.from_messages(
      [
          (
              "human",
              "You are a helpful assistant to translate English into French. Response only the translation.\n"
              "{input}",
          ),
      ]
  )

  chain = prompt | llm | StrOutputParser()

  def invoke(message):
      return chain.invoke({"input": message})

The ``llm`` model in this example uses a chat model deployed with `AI Quick Actions <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md>`_.

You can find a few example applications in the `model artifact template <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/LLM/deployment/model_artifacts>`_, including `tool calling with OCI generative AI <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/LLM/deployment/model_artifacts/exchange_rate.py>`_ and `LangGraph multi-agent example <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/LLM/deployment/model_artifacts/graph.py>`_.

Once you added your application, you can call the ``verify()`` function to test/debug it locally:

.. code-block:: python

  llm_app.verify({
      "inputs": "Hello!",
      "module": "translate.py"
  })

Note that with the default ``score.py`` template, you will invoke your application with two keys:

* ``module``: The module in the model artifact (``my_apps`` folder) containing the application to be invoked. Here we are using the ``translate.py`` example. You can specify a default module using the ``DEFAULT_MODULE`` environment variables.
* ``inputs``: the value should be the payload for your application module. This example uses a string. However, you can use list or other JSON payload for your application.

The response will have the following format:

.. code-block:: python

  {
      "outputs": "The outputs returned by invoking your app/agent",
      "error": "Error message, if any.",
      "traceback": "Traceback, if any.",
      "id": "The ID for identifying the request.",
  }

If there is an error when invoking your app/agent, the ``error`` message along with the ``traceback`` will be returned in the response.

Register the Model Artifact
***************************

Once your apps and agents are ready, you need save it to OCI Data Science Model Catalog before deployment:

.. code-block:: python3

  llm_app.save(display_name="LLM Apps", ignore_introspection=True)


Build Container Image
*********************

Before deploying the model, you will need to build a container image with the dependencies for your apps and agents.

To configure your environment for pushing image to OCI container registry (OCIR). Please refer to the OCIR documentation for `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm>`.

The `container image template <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/LLM/deployment/container>`_ contains files for building a container image for OCI Data Model Deployment service. You can add your dependencies into the ``requirement.txt`` file. You may also modify the ``Dockerfile`` if you need to add system libraries.

```bash
docker build -t <image-name:tag> .
```

Once the image is built, you can push it to OCI container registry.
```bash
docker push <image-name:tag>
```

Deploy as Model Deployment
**************************

To deploy the model, simply call the ``deploy()`` function with your settings:
* For most application, a CPU shape would be sufficient.
* Specify log group and log OCID to enable logging for the deployment.
* `Custom networking <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-create-cus-net.htm>`_ with internet access is required for accessing external APIs or OCI Generative AI APIs in a different region.
* Add environments variables as needed by your application, including any API keys or endpoints.
* You may set the ``DEFAULT_MODULE`` for invoking the default app

.. code-block:: python3

  import os

  generic_model.deploy(
      display_name="LLM Apps",
      deployment_instance_shape="VM.Standard.E4.Flex",
      deployment_log_group_id="<log_group_ocid>",
      deployment_predict_log_id="<log_ocid>",
      deployment_access_log_id="<log_ocid>",
      deployment_image="<image-name:tag>",
      # Custom networking with internet access is needed for external API calls.
      deployment_instance_subnet_id="<subnet_ocid>",
      # Add environments variables as needed by your application.
      # Following are just examples
      environment_variables={
          "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
          "PROJECT_COMPARTMENT_OCID": os.environ["PROJECT_COMPARTMENT_OCID"],
          "LLM_ENDPOINT": os.environ["LLM_ENDPOINT"],
          "DEFAULT_MODULE": "translate.py",
      }
  )

Invoking the Deployment
***********************

Once the deployment is active, you can invoke the application with HTTP requests. For example:

.. code-block:: python3

  import oci
  import requests

  response = requests.post(
      endpoint,
      json={
          "inputs": "Hello!",
      },
      auth=oci.auth.signers.get_resource_principals_signer()
  )
  response.json()

The response will be similar to the following:

.. code-block:: python3

  {
      'error': None,
      'id': 'fa3d7111-326f-4736-a8f4-ed5b21654534',
      'outputs': 'Bonjour!',
      'traceback': None
  }

Alternatively, you can use OCI CLI to invoke the model deployment. Remember to replace the ``model_deployment_url`` with the actual model deployment url, which you can find in the output from deploy step.

.. code-block:: shell

    oci raw-request --http-method POST --target-uri <model_deployment_url>/predict --request-body '{"input": "Hello!"}' --auth resource_principal
