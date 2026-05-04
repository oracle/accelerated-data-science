#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Utilities to compute rule-based meta-features for forecast model selection."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from scipy.stats import skew, kurtosis
from ads.opctl.operator.lowcode.common.const import DataColumns

EXOG_FEATURE_KEYS = (
    "exog_num_features",
    "exog_missing_ratio",
    "exog_mean_abs_mean",
    "exog_std_mean",
    "exog_skew_mean",
    "exog_last_abs_mean",
    "exog_target_corr_abs_mean",
)

FREQ_CODE = {"daily": 1, "weekly": 2, "monthly": 3, "yearly": 4}


def _safe_float(value, default=0.0):
    """Return a finite float, otherwise a default value."""
    try:
        v = float(value)
    except Exception:
        return float(default)
    if np.isnan(v) or np.isinf(v):
        return float(default)
    return v


def _sum_sq_first_n(values: np.ndarray, n: int = 5) -> float:
    """Return sum of squares of first n finite values."""
    arr = np.asarray(values, dtype=float)[:n]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return _safe_float(np.sum(arr ** 2))


def _spectral_entropy(x: np.ndarray) -> float:
    """Compute normalized spectral entropy from periodogram power."""
    if len(x) < 8:
        return 0.0
    centered = x - np.mean(x)
    ps = np.abs(np.fft.rfft(centered)) ** 2
    if ps.size <= 1:
        return 0.0
    ps = ps[1:]
    s = ps.sum()
    if s <= 0:
        return 0.0
    p = ps / s
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return _safe_float(ent)


def _hurst_exponent(x: np.ndarray) -> float:
    """Estimate Hurst exponent using variance of differenced lags."""
    n = len(x)
    if n < 20:
        return 0.5
    max_lag = min(30, n // 2)
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        if diff.size == 0:
            continue
        tau.append(np.std(diff))
    tau = np.asarray(tau, dtype=float)
    if tau.size < 4 or np.any(tau <= 0):
        return 0.5
    slope = np.polyfit(np.log(lags[:tau.size]), np.log(tau), 1)[0]
    return _safe_float(slope * 2.0, default=0.5)


def _seasonal_periods(frequency: str) -> dict:
    """Map frequency to relevant seasonal periods used for features."""
    freq = (frequency or "").lower()
    if freq == "monthly":
        return {"seasonality_m": 12}
    if freq == "weekly":
        return {"seasonality_y": 52}
    if freq == "daily":
        return {"seasonality_w": 7, "seasonality_y": 365}
    if freq == "yearly":
        return {}
    return {}


def _safe_acf_values(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return ACF values for lags 1..nlags with safe fallback."""
    if len(x) < max(3, nlags + 1):
        return np.array([], dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = acf(x, nlags=nlags, fft=True)
        return np.asarray(vals[1:], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _safe_pacf_values(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return PACF values for lags 1..nlags with safe fallback."""
    if len(x) < max(4, nlags + 2):
        return np.array([], dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = pacf(x, nlags=nlags, method="ywadjusted")
        return np.asarray(vals[1:], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _base_feature_dict() -> Dict[str, float]:
    return {
        "length": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "cv": 0.0,
        "skew": 0.0,
        "kurtosis": 0.0,
        "trend_slope": 0.0,
        "trend": 0.0,
        "linearity": 0.0,
        "curvature": 0.0,
        "spikiness": 0.0,
        "e_acf1": 0.0,
        "stability": 0.0,
        "lumpiness": 0.0,
        "entropy": 0.0,
        "hurst": 0.5,
        "nonlinearity": 0.0,
        "ur_pp": 0.0,
        "ur_kpss": 0.0,
        "adf_pvalue": 1.0,
        "y_acf1": 0.0,
        "diff1y_acf1": 0.0,
        "diff2y_acf1": 0.0,
        "y_acf5": 0.0,
        "diff1y_acf5": 0.0,
        "diff2y_acf5": 0.0,
        "y_pacf5": 0.0,
        "diff1y_pacf5": 0.0,
        "diff2y_pacf5": 0.0,
        "sediff_acf1": 0.0,
        "sediff_seacf1": 0.0,
        "sediff_acf5": 0.0,
        "seas_pacf": 0.0,
        "autocorr_lag1": 0.0,
        "autocorr_lag7": 0.0,
        "seasonality_7": 0.0,
        "seasonality_q": 0.0,
        "seasonality_m": 0.0,
        "seasonality_w": 0.0,
        "seasonality_d": 0.0,
        "seasonality_y": 0.0,
        "last_value": 0.0,
    }


def extract_meta_features(ts: pd.Series, frequency: str = None) -> dict:
    """Extract robust meta-features for model selection from one time series."""
    ts = pd.to_numeric(ts, errors="coerce").dropna()
    features = _base_feature_dict()
    if ts.empty:
        return features

    x = ts.values.astype(float)
    t = np.arange(len(ts), dtype=float)
    mean_val = ts.mean()
    std_val = ts.std()

    if len(ts) >= 2:
        lin = np.polyfit(t, x, 1)
        slope = lin[0]
        linearity = lin[0]
        fitted_lin = np.polyval(lin, t)
        resid_lin = x - fitted_lin
    else:
        slope = 0.0
        linearity = 0.0
        resid_lin = x - np.mean(x)

    if len(ts) >= 3:
        quad = np.polyfit(t, x, 2)
        fitted_quad = np.polyval(quad, t)
        ss_lin = np.sum((x - fitted_lin) ** 2) + 1e-8
        ss_quad = np.sum((x - fitted_quad) ** 2) + 1e-8
        curvature = quad[0]
        nonlinearity = max(0.0, (ss_lin - ss_quad) / ss_lin)
    else:
        curvature = 0.0
        nonlinearity = 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                adf_result = adfuller(x, autolag="AIC")
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
    except Exception:
        adf_stat = 0.0
        adf_pvalue = 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat = kpss(x, regression="c", nlags="auto")[0]
    except Exception:
        kpss_stat = 0.0

    y_acf = _safe_acf_values(x, nlags=5)
    d1 = np.diff(x, n=1)
    d2 = np.diff(x, n=2)
    d1_acf = _safe_acf_values(d1, nlags=5)
    d2_acf = _safe_acf_values(d2, nlags=5)

    y_pacf = _safe_pacf_values(x, nlags=5)
    d1_pacf = _safe_pacf_values(d1, nlags=5)
    d2_pacf = _safe_pacf_values(d2, nlags=5)

    # Chunk-wise dispersion features (FFORMA-inspired).
    window = max(8, len(x) // 10)
    n_chunks = len(x) // window
    if n_chunks >= 2:
        chunks = np.array_split(x[: n_chunks * window], n_chunks)
        chunk_means = np.array([np.mean(c) for c in chunks], dtype=float)
        chunk_vars = np.array([np.var(c) for c in chunks], dtype=float)
        stability = np.var(chunk_means)
        lumpiness = np.var(chunk_vars)
    else:
        stability = 0.0
        lumpiness = 0.0

    features.update({
        "length": float(len(ts)),
        "mean": _safe_float(mean_val),
        "std": _safe_float(std_val),
        "min": _safe_float(ts.min()),
        "max": _safe_float(ts.max()),
        "cv": _safe_float(std_val / (abs(mean_val) + 1e-8)),
        "skew": _safe_float(skew(x)),
        "kurtosis": _safe_float(kurtosis(x)),
        "trend_slope": _safe_float(slope),
        "trend": _safe_float(abs(slope) / (std_val + 1e-8)),
        "linearity": _safe_float(linearity),
        "curvature": _safe_float(curvature),
        "spikiness": _safe_float(np.var((resid_lin - np.mean(resid_lin)) ** 2)),
        "e_acf1": _safe_float(pd.Series(resid_lin).autocorr(lag=1)),
        "stability": _safe_float(stability),
        "lumpiness": _safe_float(lumpiness),
        "entropy": _safe_float(_spectral_entropy(x)),
        "hurst": _safe_float(_hurst_exponent(x), default=0.5),
        "nonlinearity": _safe_float(nonlinearity),
        "ur_pp": _safe_float(adf_stat),
        "ur_kpss": _safe_float(kpss_stat),
        "adf_pvalue": _safe_float(adf_pvalue, default=1.0),
        "y_acf1": _safe_float(y_acf[0] if y_acf.size >= 1 else 0.0),
        "diff1y_acf1": _safe_float(d1_acf[0] if d1_acf.size >= 1 else 0.0),
        "diff2y_acf1": _safe_float(d2_acf[0] if d2_acf.size >= 1 else 0.0),
        "y_acf5": _safe_float(_sum_sq_first_n(y_acf, n=5)),
        "diff1y_acf5": _safe_float(_sum_sq_first_n(d1_acf, n=5)),
        "diff2y_acf5": _safe_float(_sum_sq_first_n(d2_acf, n=5)),
        "y_pacf5": _safe_float(_sum_sq_first_n(y_pacf, n=5)),
        "diff1y_pacf5": _safe_float(_sum_sq_first_n(d1_pacf, n=5)),
        "diff2y_pacf5": _safe_float(_sum_sq_first_n(d2_pacf, n=5)),
        "autocorr_lag1": _safe_float(ts.autocorr(lag=1)),
        "autocorr_lag7": _safe_float(ts.autocorr(lag=7)),
        "seasonality_7": _safe_float(ts.autocorr(lag=7)),
        "last_value": _safe_float(ts.iloc[-1]),
    })

    periods = _seasonal_periods(frequency)
    for feature_name, p in periods.items():
        if len(x) > p:
            features[feature_name] = _safe_float(pd.Series(x).autocorr(lag=p))

            sediff = x[p:] - x[:-p]
            sediff_acf_vals = _safe_acf_values(sediff, nlags=max(5, p))
            sediff_pacf_vals = _safe_pacf_values(sediff, nlags=max(1, min(5, len(sediff) - 2)))

            features["sediff_acf1"] = _safe_float(
                sediff_acf_vals[0] if sediff_acf_vals.size >= 1 else features["sediff_acf1"]
            )
            features["sediff_seacf1"] = _safe_float(
                sediff_acf_vals[p - 1] if sediff_acf_vals.size >= p else features["sediff_seacf1"]
            )
            features["sediff_acf5"] = _safe_float(
                _sum_sq_first_n(sediff_acf_vals, n=5) if sediff_acf_vals.size else features["sediff_acf5"]
            )
            features["seas_pacf"] = _safe_float(
                sediff_pacf_vals[0] if sediff_pacf_vals.size >= 1 else features["seas_pacf"]
            )

    # Keep quarterly flag for compatibility; most datasets here do not use Q frequency.
    if len(x) > 4:
        features["seasonality_q"] = _safe_float(pd.Series(x).autocorr(lag=4))

    return features


def _compute_exog_summaries(
        df: pd.DataFrame,
        exog_cols: Iterable[str],
        target_col: str,
) -> Dict[str, float]:
    summary = {key: 0.0 for key in EXOG_FEATURE_KEYS}
    exog_cols = [col for col in exog_cols if col in df.columns]
    if not exog_cols:
        return summary

    exog = df[exog_cols].apply(pd.to_numeric, errors="coerce")
    exog_values = exog.to_numpy(dtype=float)

    summary["exog_num_features"] = float(len(exog_cols))
    summary["exog_missing_ratio"] = float(np.isnan(exog_values).mean()) if exog_values.size else 0.0
    summary["exog_mean_abs_mean"] = (
        float(np.nanmean(np.abs(exog_values))) if exog_values.size else 0.0
    )
    summary["exog_std_mean"] = (
        float(np.nanmean(exog.std(numeric_only=True).values)) if len(exog_cols) else 0.0
    )
    summary["exog_skew_mean"] = (
        float(np.nanmean(exog.skew(numeric_only=True).values)) if len(exog_cols) else 0.0
    )
    summary["exog_last_abs_mean"] = (
        float(np.nanmean(np.abs(exog.tail(1).to_numpy(dtype=float)))) if not exog.empty else 0.0
    )

    corr_vals: List[float] = []
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    for col in exog_cols:
        series = pd.to_numeric(exog[col], errors="coerce")
        valid = series.notna() & target_numeric.notna()
        if valid.sum() < 3:
            continue
        x_vals = series[valid]
        y_vals = target_numeric[valid]
        if x_vals.nunique() <= 1 or y_vals.nunique() <= 1:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = x_vals.corr(y_vals)
        if pd.notna(corr):
            corr_vals.append(abs(float(corr)))
    summary["exog_target_corr_abs_mean"] = float(np.mean(corr_vals)) if corr_vals else 0.0
    return summary


def _infer_frequency_from_dataframe(
        df: pd.DataFrame,
        timestamp_col: str,
        series_col: str,
) -> str:
    working = df[[series_col, timestamp_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working.dropna(subset=[timestamp_col], inplace=True)
    if working.empty:
        raise ValueError("No valid timestamps found to infer frequency.")

    day_deltas: List[float] = []
    for _, grp in working.groupby(series_col):
        if len(grp) < 2:
            continue
        diffs = grp[timestamp_col].sort_values().diff().dt.days.dropna()
        if not diffs.empty:
            day_deltas.append(diffs.median())
    if not day_deltas:
        raise ValueError("Insufficient temporal coverage to infer frequency.")

    median_delta = float(np.median(day_deltas))
    if median_delta <= 1.5:
        return "daily"
    if 6 <= median_delta <= 8:
        return "weekly"
    if 27 <= median_delta <= 32:
        return "monthly"
    if 360 <= median_delta <= 370:
        return "yearly"
    raise ValueError(f"Cannot map median delta {median_delta:.2f} days to a supported frequency.")


def _map_frequency_key(freq: Optional[str]) -> Optional[str]:
    if not freq:
        return None
    freq_lower = freq.lower()
    if freq_lower.startswith("d"):
        return "daily"
    if freq_lower.startswith("b"):
        return "daily"
    if freq_lower.startswith("h"):
        return "daily"
    if freq_lower.startswith("w"):
        return "weekly"
    if freq_lower.startswith("m"):
        return "monthly"
    if freq_lower.startswith("q"):
        return "monthly"
    if freq_lower.startswith("a") or freq_lower.startswith("y"):
        return "yearly"
    return None


def build_meta_features(
        df: pd.DataFrame,
        *,
        target_col: str,
        series_col: str,
        timestamp_col: Optional[str],
        horizon: Optional[float],
        frequency_hint: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Historical data is empty; cannot compute meta-features.")

    freq_key = _map_frequency_key(frequency_hint)
    if not freq_key and timestamp_col and timestamp_col in df.columns:
        try:
            freq_key = _infer_frequency_from_dataframe(df, timestamp_col, series_col)
        except ValueError:
            freq_key = "daily"
    freq_key = freq_key or "daily"
    freq_code = FREQ_CODE.get(freq_key, 0)
    horizon_value = float(horizon) if horizon is not None else 0.0

    groups = df.groupby(series_col)
    rows: List[Dict[str, float]] = []
    for series_id, grp in groups:
        ordered_grp = grp
        if timestamp_col and timestamp_col in grp.columns:
            ordered_grp = grp.sort_values(timestamp_col)
        target_series = pd.to_numeric(ordered_grp[target_col], errors="coerce").dropna()
        if target_series.empty:
            continue
        features = extract_meta_features(target_series, frequency=freq_key)
        features[DataColumns.Series] = str(series_id)
        features["frequency"] = freq_key
        features["freq_code"] = float(freq_code)
        features["horizon"] = horizon_value

        exog_candidates = [
            col
            for col in ordered_grp.columns
            if col not in {series_col, timestamp_col, target_col}
        ]
        features.update(_compute_exog_summaries(ordered_grp, exog_candidates, target_col))
        rows.append(features)

    if not rows:
        raise ValueError("Unable to compute meta-features; all series are empty after cleaning.")

    return pd.DataFrame(rows)
