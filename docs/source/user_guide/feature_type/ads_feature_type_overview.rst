Overview
********

There is a distinction between the data type of a feature and the nature of data that it represents. The data type represents the form of the data that the computer understands. ADS uses the term "feature type" to refer to the nature of the data. For example, a medical record id could be represented as an integer, its data type, but the feature type would be “medical record id”. The feature type represents the data the way the data scientist understands it. Pandas uses the term 'column' or 'Series' to refer to a column of data. In ADS the term 'feature' is used to refer to a column or series when feature types have been assigned to it.


ADS provides the feature type module on top of your Pandas dataframes and series to manage and use the typing information to better understand your data. The feature type framework comes with some common feature types.  However, the power of using feature types is that you can easily create your own and apply them to your specific data. You don’t need to try to represent your data in a synthetic way that does not match the nature of your data. This framework allows you to create methods that validate whether the data fits the specifications of your organization. For example, for a medical record type you could create methods to validate that the data is properly formatted. You can also have the system generate warnings to sure the data is valid as a whole or create graphs for summary plots.

The framework allows you to create and assign multiple feature types.  For example, a medical record id could also have a feature type id and an integer feature type.

Key Components
==============

The feature type system allows data scientists to separate the concept of how data is represented physically from what the data actually measures. That is, the data can have feature types that classify the data based on what it represents and not how the data is stored in memory. Each set of data can have multiple feature types through a system of multiple inheritances. For example, an organization that sells cars might have a set of data that represents their purchase price of a car, that is the wholesale price. You could have a feature set of ``wholesale_price``, ``car_price``, ``USD``, and ``continuous``.  This multiple inheritance allows a data scientist to create feature type warnings and feature type validators for each feature type.

A feature type is a class that inherits from ``FeatureType``. It has several attributes and methods that can be overridden to customize the properties of the feature type. The following is a brief summary of some of the key methods.  

Correlations
------------

There are also various correlation methods, such as ``.correlation_ratio()``, ``.pearson()``, and ``.cramersv()`` that provide information about the correlation between different features in the form of a dataframe. Each row represents a single correlation metric.  This information can also be represented in a plot with the ``.correlation_ratio_plot()``, ``.pearson_plot()``, and ``.cramersv_plot()`` methods.

Multiple Inheritance
--------------------

This is done through a system of inheritance. For example, a hospital may have a medical record number for each patient. That data might have the ``patient_id``, ``id``, and ``integer`` feature types. The ``patient_id`` is the child feature type with ``id`` being its parent. The ``integer`` is the parent of the ``id`` feature type. It's also the last feature type in the inheritance chain, and is called the default feature type.

When calling attributes and methods on a feature type, ADS searches the inheritance chain for the first matching feature type that defines the attribute or method that you are calling. For example, you want to produce statistics for the previously described patient id feature.  Assume that the ``patient_id`` class didn't override the ``.feature_stat()`` method. ADS would then look to the ``id`` feature type and see if it was overridden. If it was, it dispatches that method.

This system allows you to over override the methods that are specific to the feature type that you are creating and improves the reusability of your code. The default feature types are specified by ADS, and they have overridden all the attributes and methods with smart defaults.  Therefore, you don't need to override any of these properties unless you want to.

Summary Plot
------------

The ``.feature_plot()`` method returns a Seaborn plot object that summarizes the feature. You can define what you want the plot to look like for your feature. Further, you can modify the plot after it's returned, which allows you to customize it to fit your specific needs.

Summary Statistics
------------------

The ``.feature_stat()`` method returns a dataframe where each row represents a summary statistic and the numerical value for that statistic. You can customize this so that it returns summary statistics that are relevant to your specific feature type. For example, a credit card feature type may return a count of the financial network that issued the cards.

Validators
----------

The feature type validators are a set of ``is_*`` methods, where ``*`` is generally the name of the feature type. For example, the method ``.is_wholesale_price()``\ can create a boolean Pandas Series that indicates what values meet the validation criteria. It allows you to quickly identify which values need to be filtered, or require future examination into problems in the data pipeline. The feature type validators can be as complex as necessary. For example, they might take a client ID and call an API to validate each client ID is active.

Warnings
--------

Feature type warnings are used for rapid validation of the data. For example, the ``wholesale_price`` might have a method that ensures that the value is a positive number because you can’t purchase a car with negative money. The ``car_price`` feature type may have a check to ensure that it is within a reasonable price range. ``USD`` can check the value to make sure that it represents a valid US dollar amount. It can’t have values below one cent. The ``continuous`` feature type is the default feature type, and it represents the way the data is stored internally.

Forms of Feature Types
======================

There are several different forms of feature types. These are designed to balance the need to document a feature type and the ease of customization.  With each feature that you define you can specify multiple feature types.  The custom feature type gives you the most flexibility in that all the attributes and methods of the ``FeatureType`` class can be overridden. The tag feature type allows you to create a feature type that essentially is a label. Its attributes and methods cannot be overridden, but it allows you to create a feature type without creating a class. The default type is provided by ADS. It is based on the Pandas `dtype`, and sets the default attributes and methods. Each inheritance chain automatically ends in a default feature type.

Custom
-------

The most common and powerful feature type is the custom feature type. It is a Python class that inherits from ``FeatureType``. It has attributes and methods that you can be override to define the properties of the feature type to fit your specific needs. 

As with multiple inheritance, a custom feature type uses an inheritance chain to determine which attribute or method is dispatched when called. The idea is that you would have a feature that has many custom feature types with each feature type being more specific to the nature of the feature's data. Therefore, you only create the attributes and methods that are specific to the child feature type and the rest are reused from other custom or default feature types. This allows for the abstraction of the concepts that your feature represents and the reusability of your code.

Since a custom feature type is a Python class, you can add user-defined attributes and methods to the feature type to extend its capabilities.

Custom feature types must be registered with ADS before you can use them.

Default
-------

The default feature type is based on the Pandas ``dtype``. Setting the default feature type is optional when specifying the inheritance chain for a feature.  ADS automatically appends the default feature type as an ancestor to all custom feature types. The default feature type is listed before the tag feature types in the inheritance chain. Each feature only has one default feature type. You can’t mute or remove it unless the underlying Pandas ``dtype`` 
has changed. For example, you have a Pandas Series called ``series`` that has a ``dtype`` of ``string`` so its default feature type is ``string``. If you change the type by calling ``series = series.astype('category')``, then the default feature type is automatically changed to ``categorical``.

ADS automatically detects the ``dtype`` of each Series and sets the default feature type. The default feature type can be one of the following:

* ``boolean``
* ``category``
* ``continuous``
* ``date_time``
* ``integer``
* ``object``
* ``string``

This example creates a Pandas Series of credit card numbers, and prints the default feature type:

.. code-block:: python3

    series = pd.Series(["4532640527811543", "4556929308150929", "4539944650919740"], name='creditcard')
    series.ads.default_type

.. parsed-literal::

    'string'

You can include the default feature type using the ``.feature_type`` property. If you do, then the default feature type isn’t added a second time.

.. code-block:: python3

    series.ads.feature_type = ['credit_card', 'string']
    series.ads.feature_type

.. parsed-literal::

    ['credit_card', 'string']

You can't directly create or modify default feature types.

Tag
---

It's often convenient to tag a dataset with additional information without the need to create a custom feature type class. This is the role of the ``Tag()`` function, which allows you to create a feature type without having to explicitly define and register a class. The trade-off is that you can’t define most attributes and all methods of the feature type. Therefore, tools like feature type warnings and validators, and summary statistics and plots cannot be customized. 

Tags are semantic and provide more context about the actual meaning of a feature. This could directly affect the interpretation of the information.

The process of creating your tag is the same as setting the feature types because it is a feature type. You use the ``.feature_type`` property to create tags on a feature type.

The next example creates a set of credit card numbers, sets the feature type to ``credit_card``, and tags the dataset to be inactive cards.  Also, the cards are from North American financial institutions. You can put any text you want in the ``Tag()`` because no underlying feature type class has to exist.

.. code-block:: python3

    series = pd.Series(["4532640527811543", "4556929308150929", "4539944650919740", 
                        "4485348152450846"], name='Credit Card')
    series.ads.feature_type=['credit_card', Tag('Inactive Card'), Tag('North American')]
    series.ads.feature_type

.. parsed-literal::

    ['credit_card', 'string', 'Inactive Card', 'North American']

Tags are always listed after the other feature types:

A list of tags can be obtained using the ``tags`` attribute:

.. code-block:: python3

    series.ads.tags

.. parsed-literal::

    ['Inactive Card', 'North American']

