Distributed XGBoost & LightGBM
------------------------------

LightGBM
''''''''

For further examples and comprehensive documentation see `LightGBM <https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html#dask>`_ and `Github Examples <https://github.com/microsoft/LightGBM/tree/master/examples/python-guide/dask>`_

.. code-block:: python

    import os
    import joblib
    import dask.array as da
    from dask.distributed import Client
    from sklearn.datasets import make_blobs

    import lightgbm as lgb

    if __name__ == "__main__":
        print("loading data")
        size = int(os.environ.get("SIZE", 1000))
        X, y = make_blobs(n_samples=size, n_features=50, centers=2)
        client = Client(
            f"{os.environ['SCHEDULER_IP']}:{os.environ.get('SCHEDULER_PORT','8786')}"
        )

        print("distributing training data on the Dask cluster")
        dX = da.from_array(X, chunks=(100, 50))
        dy = da.from_array(y, chunks=(100,))

        print("beginning training")
        dask_model = lgb.DaskLGBMClassifier(n_estimators=10)
        dask_model.fit(dX, dy)
        assert dask_model.fitted_

        print("done training")

        # Convert Dask model to sklearn model
        sklearn_model = dask_model.to_local()
        print(type(sklearn_model)) #<class 'lightgbm.sklearn.LGBMClassifier'>
        joblib.dump(sklearn_model, "sklearn-model.joblib")



XGBoost
'''''''

For further examples and comprehensive documentation see `XGBoost <https://xgboost.readthedocs.io/en/stable/tutorials/dask.html>`_

XGBoost has a Scikit-Learn interface, this provides a familiar programming interface
that mimics the scikit-learn estimators with higher level of of abstraction. The
interface is easier to use compared to the functional interface but with more
constraints. It’s worth mentioning that, although the interface mimics scikit-learn
estimators, it doesn’t work with normal scikit-learn utilities like GridSearchCV
as scikit-learn doesn’t understand distributed dask data collection.


.. code-block:: python

  import os
  from distributed import LocalCluster, Client
  import xgboost as xgb

  def main(client: Client) -> None:
      X, y = load_data()
      clf = xgb.dask.DaskXGBClassifier(n_estimators=100, tree_method="hist")
      clf.client = client  # assign the client
      clf.fit(X, y, eval_set=[(X, y)])
      proba = clf.predict_proba(X)


  if __name__ == "__main__":
    with Client(f"{os.environ['SCHEDULER_IP']}:8786") as client:
        main(client)
