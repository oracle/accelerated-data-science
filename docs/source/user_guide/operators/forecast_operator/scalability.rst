===========
Scalability
===========

Cloud-Native
------------

Operators are fully OCI-native, allowing seamless transition from local execution to the cloud. For example, after running an operator locally:

.. code-block:: bash

    ads operator run -f forecast.yaml

You can move the execution to the cloud with a single additional flag:

.. code-block:: bash

    ads operator run -f forecast.yaml -b job


Cost and Performance Optimization
---------------------------------

Users have full control over the shape and scale of their job runs. Operators are highly parallelized, enabling efficient resource utilization. If CPU utilization is high, you can increase the number of CPUs to improve performance. Larger shapes generally run faster, and while they may appear more expensive, they often reduce overall job costs due to shorter run times.
