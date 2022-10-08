Model Schema
************

The data schema provides a definition of the format and nature of the data that the model expects. It also defines the output data from the model inference. The ``.populate_schema()`` method accepts the parameters, ``data_sample`` or ``X_sample``, and ``y_sample``. When using these parameters, the model artifact gets populates the input and output data schemas.

The ``.schema_input`` and ``.schema_output`` properties are ``Schema`` objects that define the schema of each input column and the output.  The ``Schema`` object contains these fields:

*  ``description``: Description of the data in the column.
*  ``domain``: A data structure that defines the domain of the data.  The restrictions on the data and summary statistics of its distribution.

   -  ``constraints``: A data structure that is a list of expression objects that defines the constraints of the data.

      -  ``expression``: A string representation of an expression that can be evaluated by the language corresponding to the value provided in ``language`` attribute. The default value for language is ``python``.

         -  ``expression``: Required. Use the ``string.Template`` format for specifying the expression. ``$x`` is used to represent the variable.
         -  ``language``: The default value is ``python``. Only ``python`` is supported.

   -  ``stats``: A set of summary statistics that defines the distribution of the data. These are determined using the feature type statistics as defined in ADS.
   -  ``values``: A description of the values of the data.

*  ``dtype``: Pandas data type
*  ``feature_type``: The primary feature type as defined by ADS.
*  ``name``: Name of the column.
*  ``required``: Boolean value indicating if a value is always required.

.. code-block:: yaml

   - description: Number of matching socks in your dresser drawer.
     domain:
       constraints:
       - expression: ($x <= 10) and ($x > 0)
         language: python
       - expression: $x in [2, 4, 6, 8, 10]
         language: python
       stats:
         count: 465.0
         lower_quartile: 3.2
         mean: 6.3
         median: 7.0
         sample_maximum: 10.0
         sample_minimum: 2.0
         standard_deviation: 2.5
         upper_quartile: 8.2
       values: Natural even numbers that are less than or equal to 10.
     dtype: int64
     feature_type: EvenNatural10
     name: sock_count
     required: true

Schema Model
------------

.. code-block:: json

    {
        "description": {
            "nullable": true,
            "required": false,
            "type": "string"
        },
        "domain": {
            "nullable": true,
            "required": false,
            "schema": {
            "constraints": {
                "nullable": true,
                "required": false,
                "type": "list"
            },
            "stats": {
                "nullable": true,
                "required": false,
                "type": "dict"
            },
            "values": {
                "nullable": true,
                "required": false,
                "type": "string"
            }
            },
            "type": "dict"
        },
        "dtype": {
            "nullable": false,
            "required": true,
            "type": "string"
        },
        "feature_type": {
            "nullable": true,
            "required": false,
            "type": "string"
        },
        "name": {
            "nullable": false,
            "required": true,
            "type": [
            "string",
            "number"
            ]
        },
        "order": {
            "nullable": true,
            "required": false,
            "type": "integer"
        },
        "required": {
            "nullable": false,
            "required": true,
            "type": "boolean"
        }
        }

Generating Schema
-----------------

To auto generate schema from the training data, provide X sample and the y sample while preparing the model artifact.

Eg.

.. code-block:: python3

    import tempfile
    from ads.model.framework.sklearn_model import SklearnModel
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Load dataset and Prepare train and test split 
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Train a LogisticRegression model 
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)

    # Instantite ads.model.SklearnModel using the sklearn LogisticRegression model
    sklearn_model = SklearnModel(estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    sklearn_model.prepare(inference_conda_env="dataexpl_p37_cpu_v3", X_sample=trainx, y_sample=trainy)

Calling ``.schema_input`` or ``.schema_output`` shows the schema in a YAML format.

Alternatively, you can check the ``output_schema.json`` file for the content of the schema_output:

.. code-block:: python3

    with open(path.join(path_to_artifact_dir, "output_schema.json"), 'r') as f:
        print(f.read())


.. code-block:: json

    {
        "schema": [
            {
                "dtype": "int64",
                "feature_type": "Integer",
                "name": "class",
                "domain": {
                    "values": "Integer",
                    "stats": {
                        "count": 465.0,
                        "mean": 0.5225806451612903,
                        "standard deviation": 0.5000278079030275,
                        "sample minimum": 0.0,
                        "lower quartile": 0.0,
                        "median": 1.0,
                        "upper quartile": 1.0,
                        "sample maximum": 1.0
                    },
                    "constraints": []
                },
                "required": true,
                "description": "class"
            }
        ]
    }

Update the Schema
-----------------

You can update the fields in the schema:

.. code-block:: python3

    sklearn_model.schema_output[<class name>].description = 'target variable'
    sklearn_model.schema_output[<class name>].feature_type = 'Category'

You can specify a constraint for your data using ``Expression``, and call
``evaluate`` to check if the data satisfies the constraint:

.. code-block:: python3

    sklearn_model.schema_input['col01'].domain.constraints.append(Expression('($x < 20) and ($x > -20)'))

0 is between -20 and 20, so ``evaluate`` should return ``True``:

.. code-block:: python3

    sklearn_model.schema_input['col01'].domain.constraints[0].evaluate(x=0)

.. parsed-literal::

    True



You can directly populate the schema by calling ``populate_schema()``:

.. code-block:: python3

    sklearn_model.model_artifact.populate_schema(X_sample=test.X, y_sample=test.y)

You can also load your schema from a JSON or YAML file:

.. code-block:: shell

    cat <<EOF > schema.json
    {
        "schema": [
            {
                "dtype": "int64",
                "feature_type": "Category",
                "name": "class",
                "domain": {
                    "values": "Category type.",
                    "stats": {
                    "count": 465.0,
                    "unique": 2},
                    "constraints": [
                    {"expression": "($x <= 1) and ($x >= 0)", "language": "python"},
                    {"expression": "$x in [0, 1]", "language": "python"}]},
                "required": true,
                "description": "target to predict."
            }
        ]
    }
    EOF


.. code-block:: python3

    sklearn_model.schema_output = Schema.from_file('schema.json'))



