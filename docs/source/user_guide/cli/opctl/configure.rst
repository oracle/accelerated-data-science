#################
CLI Configuration
#################

.. _configuration_prerequisites:

.. admonition:: Prerequisites

    - You have completed :doc:`ADS CLI installation <../quickstart>`


Setup default values for different options while running ``OCI Data Science Jobs`` or ``OCI Data Flow``. By setting defaults, you can avoid inputing compartment ocid, project ocid, etc.

To setup configuration run -

.. code-block:: shell

  ads opctl configure

This will prompt you to setup default ADS CLI configurations for each OCI profile defined in your OCI config. By default, all the files are generated in the ``~/.ads_ops`` folder.



``~/.ads_ops/config.ini`` will contain OCI profile defaults and conda pack related information. For example:

.. code-block::

    [OCI]
    oci_config = ~/.oci/config
    oci_profile = ANOTHERPROF

    [CONDA]
    conda_pack_folder = </local/path/for/saving/condapack>
    conda_pack_os_prefix = oci://my-bucket@mynamespace/conda_environments/

``~/.ads_ops/ml_job_config.ini`` will contain defaults for running ``Data Science Job``. Defaults are set for each profile listed in your oci config file. Here is a sample -

.. code-block::

    [DEFAULT]
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>
    subnet_id = oci.xxxx.<subnet-ocid>
    log_group_id = oci.xxxx.<log_group_ocid>
    log_id = oci.xxxx.<log_ocid>
    shape_name = VM.Standard.E2.4
    block_storage_size_in_GBs = 100

    [ANOTHERPROF]
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>
    subnet_id = oci.xxxx.<subnet-ocid>
    shape_name = VM.Standard.E2.4
    log_group_id =ocid1.loggroup.oc1.xxx.xxxxx
    log_id = oci.xxxx.<log_ocid>
    block_storage_size_in_GBs = 50


``~/.ads_ops/dataflow_config.ini`` will contain defaults for running ``Data Science Job``. Defaults are set for each profile listed in your oci config file. Here is a sample -

.. code-block::

    [DEFAULT]
    compartment_id = oci.xxxx.<compartment_ocid>
    driver_shape = VM.Standard.E2.4
    executor_shape = VM.Standard.E2.4
    logs_bucket_uri = oci://mybucket@mytenancy/dataflow/logs
    script_bucket = oci://mybucket@mytenancy/dataflow/mycode/
    num_executors = 3
    spark_version = 3.0.2
    archive_bucket = oci://mybucket@mytenancy/dataflow/archive

``~/.ads_ops/ml_pipeline.ini`` will contain defaults for running ``Data Science Pipeline``. Defaults are set for each profile listed in your oci config file. Here is a sample -

.. code-block::

    [DEFAULT]
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>

    [ANOTHERPROF]
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>

``~/.ads_ops/model_deployment_config.ini`` will contain defaults for deploying ``Data Science Model``. Defaults are set for each profile listed in your oci config file. Here is a sample -

.. code-block::

    [DEFAULT]
    compartment_id = ocid1.compartment.oc1..<unique_ID>
    project_id = ocid1.datascienceproject.oc1.iad.<unique_ID>
    shape_name = VM.Standard.E2.4
    log_group_id = ocid1.loggroup.oc1.iad.<unique_ID>
    log_id = ocid1.log.oc1.iad.<unique_ID>
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>
    bandwidth_mbps = 10
    replica = 1
    web_concurrency = 10

    [ANOTHERPROF]
    compartment_id = ocid1.compartment.oc1..<unique_ID>
    project_id = ocid1.datascienceproject.oc1.iad.<unique_ID>
    shape_name = VM.Standard.E2.4
    log_group_id = ocid1.loggroup.oc1.iad.<unique_ID>
    log_id = ocid1.log.oc1.iad.<unique_ID>
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>
    bandwidth_mbps = 20
    replica = 2
    web_concurrency = 20


``~/.ads_ops/local_backend.ini`` will contain defaults for running jobs and pipeline steps locally. While local operations do not involve connections to OCI services, default
configurations are still set for each profile listed in your oci config file for consistency. Here is a sample -

.. code-block::

    [DEFAULT]
    max_parallel_containers = 4
    pipeline_status_poll_interval_seconds = 5


    [ANOTHERPROF]
    max_parallel_containers = 4
    pipeline_status_poll_interval_seconds = 5


Generate Starter YAML
---------------------

The examples demonstrated in this section show how to generate starter YAML specification for the Data Science Job, Data Flow Application, Data Science Model Deployment and ML Pipeline services. It takes into account the config files generated within ``ads opctl configure`` operation, as well as values extracted from the environment variables.

To generate starter specification run -

.. code-block::

    ads opctl init --help

The resource type is a mandatory attribute that needs to be provided. Currently supported resource types -  `dataflow`, `deployment`, `job` and `pipeline`.
For instance to generate starter specification for the Data Science job, run -

.. code-block::

    ads opctl init job

The resulting YAML will be printed in the console. By default the ``python`` runtime will be used.


**Supported runtimes**

 - For a ``job`` - `container`, `gitPython`, `notebook`, `python` and `script`.
 - For a ``pipeline`` - `container`, `gitPython`, `notebook`, `python` and `script`.
 - For a ``dataflow`` - `dataFlow` and `dataFlowNotebook`.
 - For a ``deployment`` - `conda` and `container`.


If you want to specify a particular runtime use -

.. code-block::

    ads opctl init job --runtime-type container

Use the ``--output`` attribute to save the result in a YAML file.

.. code-block::

    ads opctl init job --runtime-type container --output job_with_container_runtime.yaml

