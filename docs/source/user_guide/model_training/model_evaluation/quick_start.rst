Quick Start
===========

Comparing Binary Classification Models
--------------------------------------

.. code-block:: python3

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    from ads.common.model import ADSModel
    from ads.common.data import ADSData
    from ads.evaluations.evaluator import ADSEvaluator

    seed = 42


    X, y = make_classification(n_samples=10000, n_features=25, n_classes=2, flip_y=0.1)


    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.30, random_state=seed)


    lr_clf = LogisticRegression(
        random_state=0, solver="lbfgs", multi_class="multinomial"
    ).fit(trainx, trainy)

    rf_clf = RandomForestClassifier(n_estimators=50).fit(trainx, trainy)

    bin_lr_model = ADSModel.from_estimator(lr_clf, classes=[0, 1])
    bin_rf_model = ADSModel.from_estimator(rf_clf, classes=[0, 1])

    evaluator = ADSEvaluator(
        ADSData(testx, testy),
        models=[bin_lr_model, bin_rf_model],
        training_data=ADSData(trainx, trainy),
    )


    print(evaluator.metrics)


Comparing Multi Classification Models
-------------------------------------

.. code-block:: python3

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    from ads.common.model import ADSModel
    from ads.common.data import ADSData
    from ads.evaluations.evaluator import ADSEvaluator

    seed = 42


    X, y = make_classification(
        n_samples=10000, n_features=25, n_classes=3, flip_y=0.1, n_clusters_per_class=1
    )


    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.30, random_state=seed)


    lr_multi_clf = LogisticRegression(
        random_state=0, solver="lbfgs", multi_class="multinomial"
    ).fit(trainx, trainy)

    rf_multi_clf = RandomForestClassifier(n_estimators=10).fit(trainx, trainy)

    multi_lr_model = ADSModel.from_estimator(lr_multi_clf)
    multi_rf_model = ADSModel.from_estimator(rf_multi_clf)


    evaluator = ADSEvaluator(
        ADSData(testx, testy),
        models=[multi_lr_model, multi_rf_model],
    )


    print(evaluator.metrics)


    
Comparing Regression Models
---------------------------

.. code-block:: python3

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.ensemble import RandomForestClassifier

    from ads.common.model import ADSModel
    from ads.common.data import ADSData
    from ads.evaluations.evaluator import ADSEvaluator

    seed = 42


    X, y = make_regression(n_samples=10000, n_features=10, n_informative=2, random_state=42)

    trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=seed)

    lin_reg = LinearRegression().fit(trainx, trainy)

    lasso_reg = Lasso(alpha=0.1).fit(trainx, trainy)


    lin_reg_model = ADSModel.from_estimator(lin_reg)
    lasso_reg_model = ADSModel.from_estimator(lasso_reg)

    reg_evaluator = ADSEvaluator(
        ADSData(testx, testy), models=[lin_reg_model, lasso_reg_model]
    )

    print(reg_evaluator.metrics)