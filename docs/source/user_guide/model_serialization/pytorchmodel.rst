.. PyTorchModel:

PyTorchModel
************

Overview
========

The ``PyTorchModel`` class in ADS is designed to allow you to rapidly get a PyTorch model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: _template/overview.rst

The following steps take your trained ``PyTorch`` model and deploy it into production with a few lines of code.

**Create a PyTorch Model**

Load a `ResNet18 <https://arxiv.org/pdf/1512.03385.pdf>`_ model and put it into evaluation mode.

.. code-block:: python3

    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

Initialize
==========

Instantiate a ``PyTorchModel()`` object with a PyTorch model. Each instance accepts the following parameters:

* ``artifact_dir: str``. Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: Callable``. Any model object generated by the PyTorch framework.
* ``properties: (ModelProperties, optional)``. Defaults to ``None``. The ``ModelProperties`` object required to save and deploy model.

.. include:: _template/initialize.rst

Summary Status
==============

.. include:: _template/summary_status.rst

.. figure:: figures/summary_status.png
   :align: center

Model Deployment
================

Prepare
-------

The prepare step is performed by the ``.prepare()`` method. It creates several customized files used to run the model after it is deployed. These files include:

* ``input_schema.json``: A JSON file that defines the nature of the features of the ``X_sample`` data. It includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``model.pt``: This is the default filename of the serialized model. It can be changed with the ``model_file_name`` attribute. By default, the model is stored in a PyTorch file. The parameter ``as_onnx`` can be used to save it in the ONNX format.
* ``output_schema.json``: A JSON file that defines the nature of the dependent variable in the ``y_sample`` data. It includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``runtime.yaml``: This file contains information that is needed to set up the runtime environment on the deployment server. It has information about which conda environment was used to train the model, and what environment should be used to deploy the model. The file also specifies what version of Python should be used.
* ``score.py``: This script contains the ``load_model()`` and ``predict()`` functions. The `load_model` function understands the format the model file was saved in, and loads it into memory. The ``.predict()`` method is used to make inferences in a deployed model. There are also hooks that allow you to perform operations before and after inference. You are able to modify this script to fit your specific needs.

To create the model artifacts, use the ``.prepare()`` method. The ``.prepare()`` method includes parameters for storing model provenance information. The PyTorch framework serialization only saves the model parameters. Thus, you must update the ``score.py`` file to construct the model class instance first before loading model parameters in the ``predict()`` function of ``score.py``.

The ``.prepare()`` method prepares and saves the ``score.py`` file, serializes the model and ``runtime.yaml`` file using the following parameters:

* ``as_onnx: (bool, optional)``: Defaults to ``False``. If ``True``, it will serialize as an ONNX model.
* ``force_overwrite: (bool, optional)``: Defaults to ``False``. If ``True``, it will overwrite existing files.
* ``ignore_pending_changes: bool``: Defaults to ``False``. If ``False``, it will ignore the pending changes in Git.
* ``inference_conda_env: (str, optional)``: Defaults to ``None``. Can be either slug or the Object Storage path of the conda environment. You can only pass in slugs if the conda environment is a Data Science service environment.
* ``inference_python_version: (str, optional)``: Defaults to ``None``. The version of Python to use in the model deployment.
* ``max_col_num: (int, optional)``: Defaults to ``utils.DATA_SCHEMA_MAX_COL_NUM``. Do not automatically generate the input schema if the input data has more than this number of features.
* ``model_file_name: (str)``: Name of the serialized model.
* ``namespace: (str, optional)``: Namespace of the OCI region. This is used for identifying which region the service environment is from when you provide a slug to the ``inference_conda_env`` or ``training_conda_env`` paramaters.
* ``training_conda_env: (str, optional)``: Defaults to ``None``. Can be either slug or object storage path of the conda environment that was used to train the model. You can only pass in a slug if the conda environment is a Data Science service environment.
* ``training_id: (str, optional)``: Defaults to value from environment variables. The training OCID for the model. Can be a notebook session or job OCID.
* ``training_python_version: (str, optional)``: Defaults to None. The version of Python used to train the model.
* ``training_script_path: str``: Defaults to ``None``. The training script path.
* ``use_case_type: str``: The use case type of the model. Use it with the ``UserCaseType`` class or the string provided in ``UseCaseType``. For example, ``use_case_type=UseCaseType.BINARY_CLASSIFICATION`` or ``use_case_type="binary_classification"``, see the ``UseCaseType`` class to see all supported types.
* ``X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]``: Defaults to ``None``. A sample of the input data. It is used to generate the input schema.
* ``y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]``: Defaults to None. A sample of output data. It is used to generate the output schema.
* ``**kwargs``:
    - ``dynamic_axes: (dict, optional)``: Defaults to ``None``. Optional in ONNX serialization. Specify axes of tensors as dynamic (i.e. known only at run-time).
    - ``input_names: (List[str], optional)``: Defaults to ``["input"]``. Optional in an ONNX serialization.  It is an ordered list of names to assign to the input nodes of the graph.
    - ``onnx_args: (tuple or torch.Tensor, optional)``: Required when ``as_onnx=True`` in an ONNX serialization. Contains model inputs such that ``onnx_model(onnx_args)`` is a valid invocation of the model.
    - ``output_names: (List[str], optional)``: Defaults to ``["output"]``. Optional in an ONNX serialization. It is an ordered list of names to assign to the output nodes of the graph.

Verify
------

.. include:: _template/verify.rst

* ``data: Any``: Data expected by the predict API in the ``score.py`` file. For the PyTorch serialization method, ``data`` can be in type dict, str, list, np.ndarray, or ``torch.tensor``. For the ONNX serialization method, ``data`` has to be JSON serializable or ``np.ndarray``.


Save
----

.. include:: _template/save.rst

Deploy
------

.. include:: _template/deploy.rst

Predict
-------

.. include:: _template/predict.rst

* ``data: Any``: Data expected by the predict API in the ``score.py`` file. For the PyTorch serialization method, ``data`` can be in type dict, str, list, np.ndarray, or ``torch.tensor``. For the ONNX serialization method, ``data`` has to be JSON serializable or ``np.ndarray``.

Load
====

You can restore serialization models from model artifacts, from model deployments or from models in the model catalog. This section provides details on how to restore serialization models.

.. include:: _template/loading_model_artifact.rst

.. code-block:: python3

    from ads.model.framework.pytorch_model import PyTorchModel

    model = PyTorchModel.from_model_artifact(
                    uri="/folder_to_your/artifact.zip",
                    model_file_name="model.pt",
                    artifact_dir="/folder_store_artifact"
                )

.. include:: _template/loading_model_catalog.rst

.. code-block:: python3

    from ads.model.framework.pytorch_model import PyTorchModel

    model = PyTorchModel.from_model_catalog(model_id="<model_id>",
                                            model_file_name="model.pt",
                                            artifact_dir=tempfile.mkdtemp())

.. include:: _template/loading_model_deployment.rst

.. code-block:: python3

    from ads.model.generic_model import PyTorchModel

    model = PyTorchModel.from_model_deployment(
        model_deployment_id="<model_deployment_id>"",
        model_file_name="model.pkl",
        artifact_dir=tempfile.mkdtemp())

Delete a Deployment
===================

.. include:: _template/delete_deployment.rst

Example
=======

.. code-block:: python3

    import tempfile
    import torchvision
    from ads.catalog.model import ModelCatalog
    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.pytorch_model import PyTorchModel

    # Load the PyTorch Model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # Prepare the model
    artifact_dir = tempfile.mkdtemp()
    pytorch_model = PyTorchModel(model, artifact_dir=artifact_dir)
    pytorch_model.prepare(
        inference_conda_env="generalml_p37_cpu_v1",
        training_conda_env="generalml_p37_cpu_v1",
        use_case_type=UseCaseType.IMAGE_CLASSIFICATION,
        as_onnx=False,
        force_overwrite=True,
    )

    # Update ``score.py`` by constructing the model class instance first.
    added_line = """
    import torchvision
    the_model = torchvision.models.resnet18()
    """
    with open(artifact_dir + "/score.py", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(added_line.rstrip('\r\n') + '\n' + content)

    # test_data will need to be defined based on the image requirements of ResNet18

    # Deploy the model, test it and clean up.
    pytorch_model.verify(test_data)
    model_id = pytorch_model.save()
    pytorch_model.deploy()
    pytorch_model.predict(test_data)
    pytorch_model.delete_deployment(wait_for_completion=True)
    pytorch_model.delete()

