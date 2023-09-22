.. _Release Notes:

=============
Release Notes
=============

1.0.3
-----
.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - SERVICE_VERSION
        - 0.1.256.master
        -
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        -


Release notes: September 22, 2023

* [FEATURE] Addition of ``featurestore_dataset``  as optional parameter in GenericModel ``save`` function.
* [FEATURE] Addition of ``transformation_kwargs`` in ``Transformation`` entity.
* [FEATURE] Addition of partition keys in ``FeatureGroup`` and ``Dataset``
* [FEATURE] as_of interface for time travel
* [FEATURE] Manual association of feature groups in ``Dataset`` construct and support for complex queries
* [FEATURE] Simplify the ads init experience without users need to specify the feature store endpoint
* [FEATURE] Visualisation of feature statistics with ``to_viz`` in ``Statistics`` entity
* [FIX] Validation of model ids when associated with ``Dataset``
* [UI] Stats visualisation of feature group and dataset.
* [UI] Transformation listing in the transformation tab
* [UI] Global search for feature store entities.
* [UI] Addition of onboarding page for feature store.
* [UI] Redirection of entities within the lineage tab of feature group and dataset.

1.0.2
-----
.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - Conda pack
        - `https://objectstorage.us-ashburn-1.oraclecloud.com/n/bigdatadatasciencelarge/b/service-conda-packs-fs/o/service_pack/cpu/PySpark_3.2_and_Feature_Store/1.0/fspyspark32_p38_cpu_v1#conda`
        -
      * - SERVICE_VERSION
        - 0.1.225.master
        -
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        - Par link expires Jan 5, 2026

Release notes: July 18, 2023

* [FEATURE] Supporting for deployment in ``us-ashburn`` and ``uk-london`` region.
* [FEATURE] For ``ValidationOutput`` instance, addition of ``to_summary()`` method  for validation summary details.
* [FEATURE] For ``ValidationOutput`` instance, addition of ``to_pandas()`` method  for validation detailed report.
* [FIX] Fixed unit test integration to support the merging of ADS into the main branch.
* [DOCS] For ``ValidationOutput`` instance, addition of ``to_summary()`` method  for validation summary details.
* [DOCS] For ``ValidationOutput`` instance, addition of ``to_pandas()`` method  for validation detailed report.

1.0.1
-----

.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - Conda pack
        - `https://objectstorage.us-ashburn-1.oraclecloud.com/n/bigdatadatasciencelarge/b/service-conda-packs-fs/o/service_pack/cpu/PySpark_3.2_and_Feature_Store/1.0/fspyspark32_p38_cpu_v1#conda`
        -
      * - SERVICE_VERSION
        - 0.1.218.master
        -
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        - Par link expires Jan 5, 2026


Release notes: July 5, 2023

* [FEATURE] Supporting Offline Feature Type COMPLEX
* [FEATURE] Added k8 default version as v1.25.4
* [FEATURE] Improvements in logging during materialisation of feature group and dataset and showcasing validation results during materialisation
* [FIX] Fixed creation of singleton spark session without metastore id
* [DOCS] Data Type update for Offline Feature Type COMPLEX
* [DOCS] Updated terraform default version as 1.1.x

1.0.0
----

.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - Conda pack
        - `https://objectstorage.us-ashburn-1.oraclecloud.com/n/bigdatadatasciencelarge/b/service-conda-packs-fs/o/service_pack/cpu/PySpark_3.2_and_Feature_Store/1.0/fspyspark32_p38_cpu_v1#conda`
        -
      * - SERVICE_VERSION
        - 0.1.209.master
        -
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        - Par link expires Jan 5, 2026

Release notes: June 15, 2023

* [FEATURE] Included ``FeatureStore``, ``FeatureGroup``, ``Dataset``, ``Entity`` and ``Transformation`` concepts for feature store.
* [DOCS] Included documentation for ``FeatureStore``, ``FeatureGroup``, ``Dataset``, ``Entity`` and ``Transformation`` constructs
