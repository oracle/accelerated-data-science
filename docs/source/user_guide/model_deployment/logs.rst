Logs
****

The model deployment process creates a set of workflow logs. Optionally, you can also configure the Logging service to capture access and predict logs.

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.from_id()``.

Access/Predict
==============

The ``.show_logs()`` and ``.logs()`` methods in the ``ModelDeployment`` class exposes the predict and access logs. The parameter ``log_type`` accepts ``predict`` and ``access`` to specify which logs to return. When it's not specified, the access logs are returned. The parameters ``time_start`` and ``time_end`` restrict the logs to time periods between those entries. The ``limit`` parameter limits the number of log entries that are returned.

Logs are not collected in real-time. Therefore, it is possible that logs have been emitted by the model deployment but are not currently available with the ``.logs()`` and ``.show_logs()`` methods.

``logs``
--------

This method returns a list of dictionaries where each element of the list is a log entry. Each element of the dictionary is a key-value pair from the log.

.. code-block:: python3

    deployment.logs(log_type="access", limit=10)

``show_logs``
-------------

This method returns a dataframe where each row represents a log entry. 

.. code-block:: python3

    deployment.show_logs(log_type="access", limit=10)

``Watch``
---------

You can stream the predict and access log of a model deployment using the ``.watch()`` method of a ``ModelDeployment`` object.

.. code-block:: python3

   deployment.watch() # stream predict and access log
   deployment.watch(log_type="access") # stream access log


