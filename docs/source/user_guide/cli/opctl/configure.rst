CLI Configuration
=================

**Prerequisite**

* You have completed :doc:`ADS CLI installation <../quickstart>` 


Setup default values for different options while running ``OCI Data Sciecne Jobs`` or ``OCI DataFlow``. By setting defaults, you can avoid inputing compartment ocid, project ocid, etc.

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
    shape_name = VM.Standard2.2
    block_storage_size_in_GBs = 100

    [ANOTHERPROF]
    compartment_id = oci.xxxx.<compartment_ocid>
    project_id = oci.xxxx.<project_ocid>
    subnet_id = oci.xxxx.<subnet-ocid>
    shape_name = VM.Standard2.1
    log_group_id =ocid1.loggroup.oc1.xxx.xxxxx
    log_id = oci.xxxx.<log_ocid>
    block_storage_size_in_GBs = 50


``~/.ads_ops/dataflow_config.ini`` will contain defaults for running ``Data Science Job``. Defaults are set for each profile listed in your oci config file. Here is a sample - 

.. code-block::

    [MYTENANCYPROF]
    compartment_id = oci.xxxx.<compartment_ocid>
    driver_shape = VM.Standard2.1
    executor_shape = VM.Standard2.1
    logs_bucket_uri = oci://mybucket@mytenancy/dataflow/logs
    script_bucket = oci://mybucket@mytenancy/dataflow/mycode/
    num_executors = 3
    spark_version = 3.0.2
    archive_bucket = oci://mybucket@mytenancy/dataflow/archive
