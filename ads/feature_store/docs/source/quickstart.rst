Quick start
************
1. Create a `Data Science notebook session <https://docs.oracle.com/en-us/iaas/data-science/using/create-notebook-sessions.htm>`__ to access jupyterlab interface.

2. Open a terminal in the notebook session, and then install the ``fspyspark32_p38_cpu_v2`` plugin:

  ..  code-block:: shell

     odsc conda install -s fspyspark32_p38_cpu_v3

3. Download the notebook examples from the example notebook section.

.. seealso::
   Refer to :ref:`Notebook Examples` which contains more examples for using feature store.

4. Upload the notebook in the notebook session, and then run the notebook after replacing the required variables.



**Feature Store and Data Science Concepts:**

- Getting started with  `OCI Data Science Jobs <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-about.htm>`__
- Getting started with  `Oracle Accelerated Data Science SDK <https://accelerated-data-science.readthedocs.io/en/latest/index.html>`__ to simplify `creating <https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/data_science_job.html#define-a-job>`__ and `running <https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/data_science_job.html#run-a-job-and-monitor-outputs>`__ Jobs
- Getting started with  `Data Science Environments <https://docs.oracle.com/en-us/iaas/data-science/using/conda_environ_list.htm>`__
- Getting started with  `Custom Conda Environments <https://docs.oracle.com/en-us/iaas/data-science/using/conda_create_conda_env.htm>`__

**Authentication and Policies:**

- Getting started with `OCI Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`__
- `API Key-Based Authentication <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm#sdk_authentication_methods_api_key>`__ - ``api_key``
- `Resource Principal Authentication <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm#sdk_authentication_methods_resource_principal>`__ - ``resource_principal``
- `Instance Principal Authentication <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm#sdk_authentication_methods_instance_principaldita>`__ - ``instance_principal``

.. seealso::

   Review the `Terraform section <https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/index.html#document-terraform>`__ for setting up feature store server.

.. warning::

   1. Feature store doesn’t allow parallel execution of similar logical constructs. Creation is sequential.
   2. If a failure occurs, processing stops, and rollback can’t happen. Retrying the operation isn’t supported.
   3. Define exactly one feature store construct in the YAML file. Creation of multiple feature store constructs in the YAML file causes a failure.
   4. To allow references in the feature store definition, the name of the defined logical constructs in the YAML file must be unique.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.feature_group_expectation import Expectation, Rule, ExpectationType, ValidationEngineType
    from ads.feature_store.feature_store import FeatureStore
    from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
    from ads.feature_store.transformation import TransformationMode
    import ads

    compartment_id = "ocid1.compartment.<unique_id>"
    metastore_id = "ocid1.datacatalogmetastore.oc1.iad.<unique_id>"
    api_gateway_endpoint = "https://**.{region}.oci.customer-oci.com/20230101"
    os.environ["OCI_FS_SERVICE_ENDPOINT"] = api_gateway_endpoint

    ads.set_auth(auth="api_key")

    # step1: Create the feature store
    feature_store_resource = (
        FeatureStore()
        .with_description("<feature_store_description>")
        .with_compartment_id(compartment_id)
        .with_name("<name>")
        .with_offline_config(metastore_id=metastore_id)
    )

    feature_store = feature_store_resource.create()
    entity = feature_store.create_entity(name="product")


    # step2: Create the feature store
    def transactions_df(dataframe, **kwargs):
        columns = kwargs.get('columns', '*')  # The default is to select all columns if 'columns' isn't provided
        where_clause = kwargs.get('where_clause', '')  # The default is to an empty where clause if 'where_clause' not provided

        sql_query = f"""
                    SELECT
                        {columns}
                    FROM
                        {table_name}
                    {where_clause}
                """
        return sql_query

    transformation = feature_store.create_transformation(
        transformation_mode=TransformationMode.SQL,
        source_code_func=transactions_df
    )


    # step3: Create the expectation
    expectation_suite = ExpectationSuite(expectation_suite_name="feature_definition")
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date"}
        )
    )

    input_feature_details = [FeatureDetail("rule_name").with_feature_type(FeatureType.STRING).with_order_number(1)]

    # step4: Create the statistics configuration
    stats_config = StatisticsConfig().with_is_enabled(False)

    # step5: Create the feature group
    transformation_args = {"columns": "col1, col2", "where_clause": "col3 > 100"}
    feature_group = entity.create_feature_group(
                        primary_keys=["name"],
                        partition_keys=["name"],
                        input_feature_details,
                        expectation_suite=expectation_suite,
                        expectation_type=ExpectationType.LENIENT,
                        statistics_config=stats_config,
                        name="<feature_group_name>",
                        transformation_id=transformation.id,
                        transformation_kwargs=transformation_args
                    )


  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.feature_store_registrar import FeatureStoreRegistrar

    yaml_string = """
    apiVersion: 20230101
    kind: featureStore
    spec:
      name: *feature_store_name
      offlineConfig:
        metastoreId: *metastore_id

      entity: &entity
        - kind: entity
          spec:
            name: *entity_name


      transformation: &transformation
        - kind: transformation
          spec:
            name: *transformation_name
            transformationMode: *transformation_mode
            sourceCode: *source_code

      featureGroup:
        - kind: featureGroup
          spec:
            name: *feature_group_name
            dataSource: *ds
            description: *feature_group_desc
            transformation: *transformation
            entity: *entity
            primaryKeys:
              *fg_primary_key
            inputFeatureDetails:
              - name: *feature_name
                featureType: *feature_type
                orderNumber: 1

      dataset:
        - kind: dataset
          spec:
            name: *dataset_name
            entity: *entity
            datasetIngestionMode: *ingestion_mode
            description: *dataset_description
            query: *query_statement
    """

    feature_registrar = FeatureStoreRegistrar.from_yaml(yaml_string)
