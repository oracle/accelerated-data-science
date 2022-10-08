Writing Dask Code
-----------------

Dask Integrates at many levels into the Python ecosystem.


**Run parallel computation using dask.distributed and Joblib**


Joblib can use Dask as the backend. In the following example the long running function is
distributed across the Dask cluster.

.. code-block:: python

  import time
  import joblib

  def long_running_function(i):
      time.sleep(.1)
      return i

This function can be called under Dask as a dask task which will be scheduled automatically by
Dask across the cluster. Watching the cluster utilization will show the tasks run on the
workers.

.. code-block:: python

  with joblib.parallel_backend('dask'):
      joblib.Parallel(verbose=100)(
          joblib.delayed(long_running_function)(i)
          for i in range(10))

**Run parallel computation using Scikit-Learn & Joblib**

To use the Dask backend to Joblib you have to create a Client, and wrap your code with the
``joblib.parallel_backend('dask')`` context manager.

.. code-block:: python

  import os
  from dask.distributed import Client
  import joblib

  # the cluster once created will make available the IP address of the Dask scheduler
  # through the SCHEDULER_IP environment variable
  client = Client(f"{os.environ['SCHEDULER_IP']}:8786")

  with joblib.parallel_backend('dask'):
      # Your scikit-learn code

A full example showing scaling out CPU-bound workloads; workloads with datasets
that fit in RAM, but have many individual operations that can be done in parallel.
To scale out to RAM-bound workloads (larger-than-memory datasets) use one of the
``dask-ml`` provided parallel estimators, or the dask-ml wrapped ``XGBoost`` &
``LightGBM`` estimators.

.. code-block:: python

  import numpy as np
  from dask.distributed import Client

  import joblib
  from sklearn.datasets import load_digits
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.svm import SVC

  client = Client(f"{os.environ['SCHEDULER_IP']}:8786")

  digits = load_digits()

  param_space = {
      'C': np.logspace(-6, 6, 13),
      'gamma': np.logspace(-8, 8, 17),
      'tol': np.logspace(-4, -1, 4),
      'class_weight': [None, 'balanced'],
  }

  model = SVC(kernel='rbf')
  search = RandomizedSearchCV(model, param_space, cv=3, n_iter=50, verbose=10)

  with joblib.parallel_backend('dask'):
      search.fit(digits.data, digits.target)
