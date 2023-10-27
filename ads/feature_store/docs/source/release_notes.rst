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
      * - Conda pack
        - `fs_pyspark32_p38_cpu_v1`
        -
      * - SERVICE_VERSION
        - 0.1.256.master
        -
      * - ADS_VERSION
        - oracle-ads==2.9.0rc0
        - `https://github.com/oracle/accelerated-data-science/releases/tag/v2.9.0rc0`
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        -


Release notes: September 22, 2023

* [FEATURE] Addition of ``featurestore_dataset``  as optional parameter in GenericModel ``save`` function.
* [FEATURE] Addition of ``transformation_kwargs`` in ``Transformation`` entity.
* [FEATURE] Addition of partition keys in ``FeatureGroup`` and ``Dataset``.
* [FEATURE] Addition of the as_of interface for time travel.
* [FEATURE] Manual association of feature groups in the ``Dataset`` construct and support for complex queries.
* [FEATURE] Simplify the ``ads init`` experience so you don't have to specify the feature store endpoint.
* [FEATURE] Visualisation of feature statistics with ``to_viz`` in ``Statistics`` entity.
* [FIX] Validation of model IDs when associated with ``Dataset``.
* [UI] Statistics visualisation of feature group and dataset.
* [UI] Transformation listing in the transformation tab.
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
        - `fs_pyspark32_p38_cpu_v1`
        -
      * - SERVICE_VERSION
        - 0.1.225.master
        -
      * - Terraform Stack
        - `link <https://objectstorage.us-ashburn-1.oraclecloud.com/p/vZogtXWwHqbkGLeqyKiqBmVxdbR4MK4nyOBqDsJNVE4sHGUY5KFi4T3mOFGA3FOy/n/idogsu2ylimg/b/oci-feature-store/o/beta/terraform/feature-store-terraform.zip>`__
        - Par link expires Jan 5, 2026

Release notes: July 18, 2023

* [FEATURE] Support for deployment in the ``us-ashburn`` and ``uk-london`` regions.
* [FEATURE] For a ``ValidationOutput`` instance, the addition of the ``to_summary()`` method for validation summary details.
* [FEATURE] For a ``ValidationOutput`` instance, the addition of the ``to_pandas()`` method for validation detailed report.
* [FIX] Fixed unit test integration to support the merging of ADS into the main branch.
* [DOCS] For ``ValidationOutput`` instance, the addition of the ``to_summary()`` method for validation summary details.
* [DOCS] For ``ValidationOutput`` instance, the addition of the ``to_pandas()`` method for validation detailed report.

1.0.1
-----

.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - Conda pack
        - `fs_pyspark32_p38_cpu_v1`
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
-----

.. note::

    .. list-table::
      :header-rows: 1

      * - Package Name
        - Latest Version
        - Notes
      * - Conda pack
        - `fs_pyspark32_p38_cpu_v1`
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
