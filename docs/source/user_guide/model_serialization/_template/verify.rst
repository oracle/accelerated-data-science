If you update the ``score.py`` file included in a model artifact, you can verify your changes, without deploying the model. With the ``.verify()`` method, you can debug your code without having to save the model to the model catalog and then deploying it. The ``.verify()`` method takes a set of test parameters and performs the prediction by calling the ``predict()`` function in ``score.py``. It also runs the ``load_model()`` function to load the model.

The ``verify()`` method tests whether the ``.predict()`` API works in the local environment and it takes the following parameter:

