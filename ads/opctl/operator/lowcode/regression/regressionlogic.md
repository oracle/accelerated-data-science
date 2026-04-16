
• Overview

  The regression operator builds a single sklearn Pipeline with two stages: preprocessor and regressor, then fits that pipeline on training data in ads/opctl/operator/
  lowcode/regression/model/shared_model.py:21. After training, it computes metrics, feature importance, and optionally SHAP explanations.

  Preprocessing

  Preprocessing is configured in ads/opctl/operator/lowcode/regression/operator_config.py:30 and ads/opctl/operator/lowcode/regression/operator_config.py:69. By
  default:

  - preprocessing.enabled = True
  - missing_value_imputation = True
  - categorical_encoding = True

  The actual logic is in ads/opctl/operator/lowcode/regression/model/base_model.py:130 and ads/opctl/operator/lowcode/regression/model/base_model.py:150.

  How it works:

  - It first decides which columns are numeric vs categorical.
  - If spec.column_types is provided, that overrides automatic detection.
  - Otherwise numeric pandas dtypes are treated as numeric, everything else as categorical.
  - Numeric columns get median imputation if enabled.
  - Categorical columns get most-frequent imputation if enabled.
  - Categorical columns then get one-hot encoded if enabled.
  - Anything not listed as a feature is dropped by the ColumnTransformer.

  Example:

  Suppose your features are:

  age        numeric
  income     numeric
  city       categorical
  segment    categorical

  And your training rows are:

  age  income   city   segment
  35   90000    NYC    A
  NaN  70000    SF     B
  42   NaN      NYC    NaN

  With default preprocessing, this becomes roughly:

  age   income   city_NYC   city_SF   segment_A   segment_B
  35    90000    1          0         1           0
  38.5  70000    0          1         0           1
  42    80000    1          0         1           0

  Why those filled values?

  - age missing value becomes median of numeric column.
  - income missing value becomes median of numeric column.
  - segment missing value becomes most frequent category.
  - city and segment become one-hot columns.

  Feature names are then extracted from the fitted preprocessor in ads/opctl/operator/lowcode/regression/model/base_model.py:192. That matters because downstream
  feature importance and SHAP operate on transformed columns, not always the original raw columns.

  Feature Importance

  The logic is in ads/opctl/operator/lowcode/regression/model/base_model.py:200.

  It uses three strategies, in this order:

  1. If the estimator exposes feature_importances_, use that.
  2. Else if the estimator exposes coef_, use abs(coef_).
  3. Else use permutation importance.

  That means:

  - Random forest and XGBoost usually use built-in tree importance.
  - Linear regression uses absolute coefficient magnitude.
  - KNN falls back to permutation importance because it has neither feature_importances_ nor coef_.

  Example 1: Linear regression

  If the fitted transformed model has coefficients like:

  age         2.5
  income      0.01
  city_NYC    1.2
  city_SF    -0.4

  The operator stores importance as absolute value:

  age         2.5
  city_NYC    1.2
  city_SF     0.4
  income      0.01

  So this is coefficient magnitude, not causal importance.

  Example 2: KNN permutation importance

  For KNN, the operator repeatedly shuffles one input feature at a time and measures how much model quality drops, using negative RMSE as the scoring function in ads/
  opctl/operator/lowcode/regression/model/base_model.py:210.

  If shuffling age hurts performance a lot, age gets high importance.
  If shuffling city_SF barely changes predictions, it gets low importance.

  Important nuance:

  - For tree and linear models, feature importance is computed on transformed features.
  - With one-hot encoding, one raw categorical column becomes many derived columns such as city_NYC, city_SF, city_LA.
  - So the importance table is per transformed feature, not automatically grouped back to the original raw column.

  SHAP Explainability

  The logic is in ads/opctl/operator/lowcode/regression/model/base_model.py:241.

  It only runs when generate_explanations: true.

  What it does:

  - Takes up to 200 sampled training rows.
  - Applies the fitted preprocessor to those rows.
  - Uses SHAP on the transformed data.
  - Produces:
      - global_explanations_df: average absolute SHAP value per feature
      - local_explanations_df: SHAP values for each sampled row

  There are two SHAP paths:

  - Tree models: if the estimator has feature_importances_, it uses shap.TreeExplainer in ads/opctl/operator/lowcode/regression/model/base_model.py:258
  - Other models: it uses generic shap.Explainer(model.predict, x_sample) in ads/opctl/operator/lowcode/regression/model/base_model.py:262

  How to read SHAP:

  - A SHAP value per feature per row says how much that feature pushed the prediction up or down relative to a baseline.
  - Positive SHAP means it increased the prediction.
  - Negative SHAP means it decreased the prediction.
  - Larger absolute value means stronger influence.

  Example:

  Suppose for one sampled row the model predicts house price 320000, and baseline prediction is 250000.

  SHAP for that row might look like:

  sqft          +50000
  location_A    +30000
  age           -10000
  garage        +0

  Interpretation:

  - sqft pushed price up by 50k
  - location_A pushed it up by 30k
  - age pushed it down by 10k

  The global explanation then averages abs(SHAP) across sampled rows:

  sqft          42000
  location_A    18000
  age           12000
  garage         3000

  That becomes the overall SHAP importance ranking in ads/opctl/operator/lowcode/regression/model/base_model.py:271.

  How Preprocessing Affects Importance and SHAP

  This is the most important practical point.

  Because the model is trained on transformed features:

  - feature importance is usually over transformed columns
  - SHAP is also over transformed columns
  - categorical columns become one-hot features
  - imputed values are what the model actually sees

  Example:

  Raw feature:

  city

  After preprocessing:

  city_NYC
  city_SF
  city_LA

  So you may see importance/SHAP for city_NYC rather than a single combined city.

  That means if you want a human-friendly view of raw-column importance, you would need an extra aggregation layer that sums or groups transformed columns back to their
  source feature. The current implementation does not do that.

  What the Training Flow Looks Like End-to-End

  The pipeline flow is in ads/opctl/operator/lowcode/regression/model/shared_model.py:21:

  1. Build preprocessor.
  2. Build estimator.
  3. Fit Pipeline(preprocessor, regressor).
  4. Predict on train and optional test.
  5. Compute metrics.
  6. Compute feature importance.
  7. Optionally compute SHAP.

  So the preprocessing is not a separate offline step. It is part of the fitted pipeline and is applied consistently during training, prediction, feature importance
  fallback, and SHAP sampling.

  Caveats

  - Linear-model importance uses absolute coefficients, which can be misleading if features are on very different scales.
  - Tree-model feature_importances_ is model-native importance, not SHAP.
  - KNN importance uses permutation importance, which is slower but more model-agnostic.
  - SHAP currently samples at most 200 training rows, so it is approximate for large datasets.
  - SHAP and feature importance are both tied to transformed features after preprocessing.

  If you want, I can also walk through one concrete example using a small CSV and show exactly what feature_importance.csv, global_explanations.csv, and
  local_explanations.csv would look like.