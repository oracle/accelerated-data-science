# Regression Operator

The Regression Operator trains tabular regression models with a YAML-based interface and writes consistent artifacts including predictions, metrics, feature importance, explainability outputs, and an HTML report.

## Quickstart

1. Generate starter configs:

```bash
ads operator init -t regression --overwrite --output ~/regression/
```

2. Verify the generated operator config:

```bash
ads operator verify -f ~/regression/regression.yaml
```

3. Run locally:

```bash
ads operator run -f ~/regression/regression.yaml -b local
```

## Supported Models

- `linear_regression`
- `random_forest`
- `knn`
- `xgboost`
- `auto`
