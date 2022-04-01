Predict
=======

Predictions can be made by calling the HTTP endpoint associated with the model
deployment. The ``ModelDeployment`` object ``url`` attribute 
specifies the endpoint. You could also use the
``ModelDeployment`` object with the ``.predict()`` method. The format of
the data that is passed to the HTTP endpoint depends on the setup of
the model artifact. The default setup is to pass in a Python dictionary
that has been converted to a JSON data structure. The first level
defines the feature names. The second level uses an identifier for the
observation (for example, row in the dataframe), and the value associated with
it. Assuming the model has features F1, F2, F3, F4, and F5, then 
the observations are identified by the values 0, 1, and 2 and the data would look like this:

===== == == == == ==
Index F1 F2 F3 F4 F5
===== == == == == ==
0     11 12 13 14 15
1     21 22 23 24 25
2     31 32 33 34 35
===== == == == == ==

The Python dictionary representation would be:

.. code:: ipython3

   test = { 
      'F1': { 0: 11, 1: 21, 2: 31},
      'F2': { 0: 12, 1: 22, 2: 32},
      'F3': { 0: 13, 1: 23, 2: 33},
      'F4': { 0: 14, 1: 24, 2: 34},
      'F5': { 0: 15, 1: 25, 2: 35}
   }


You can use the ``ModelDeployment`` object to call the HTTP endpoint. The returned
result is the predictions for the three observations.

.. code:: ipython3

    deployment.predict(test)


.. parsed-literal::

    {'prediction': [0, 2, 0]}


