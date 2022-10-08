====
Dask
====


Dask is a flexible library for parallel computing in Python. The documentation will
split between the two areas of writing distributed training using the ``Dask`` framework and
creating both the container and ``yaml`` spec to run the distributed workload.


.. admonition:: Dask

  This is a good choice when you want to use ``Scikit-Learn``, ``XGBoost``, ``LightGBM`` or have
  data parallel tasks for very large datasets where the data can be partitioned. 


.. toctree::
    :maxdepth: 3

    creating
    coding
    gbm
    tls
    tuning
    dashboard
