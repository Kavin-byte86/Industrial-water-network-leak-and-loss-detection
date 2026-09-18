"""
Shared feature construction for the leak-detection classifier.

The v2 Stage-2 model is trained on temporal lag / deviation / production-relative
features. Training used to build these inline, so inference had no way to produce
them and quietly substituted zeros - which silently degrades recall. Both sides
now call the same function from here.
"""

import pandas as pd

SENSORS = [f"J{i}" for i in range(1, 17)]
ROLLING_WINDOW = 12  # 12 x 5 min = 1 hour


def advanced_feature_names(available_columns=None):
    """Names of every column `add_advanced_features` produces.

    `available_columns` gates the residual-derived names, which only exist once
    Stage 1 has run.
    """
    names = []
    for s_id in SENSORS:
        names += [
            f"flow_{s_id}_lag1",
            f"flow_{s_id}_lag3",
            f"flow_{s_id}_lag6",
            f"rolling_median_flow_{s_id}",
            f"short_term_dev_flow_{s_id}",
            f"flow_per_prod_{s_id}",
        ]
        if available_columns is None or f"ml_flow_residual_{s_id}" in available_columns:
            names.append(f"residual_per_prod_{s_id}")
    return names


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal lag, short-term deviation and production-relative features.

    Rows must be contiguous and in chronological order: the lag and rolling terms
    are computed within `df` itself. A single-row frame therefore has no real
    history and its lags collapse onto the current value (via bfill), which is the
    documented degraded case - prefer passing a window of recent telemetry.
    """
    df_out = df.copy()

    # 1. Temporal lags (1 = 5 min, 3 = 15 min, 6 = 30 min)
    new_cols = {}
    for s_id in SENSORS:
        flow = df_out[f"flow_{s_id}"]
        new_cols[f"flow_{s_id}_lag1"] = flow.shift(1)
        new_cols[f"flow_{s_id}_lag3"] = flow.shift(3)
        new_cols[f"flow_{s_id}_lag6"] = flow.shift(6)

        rolling_median = flow.rolling(window=ROLLING_WINDOW, min_periods=1).median()
        new_cols[f"rolling_median_flow_{s_id}"] = rolling_median
        new_cols[f"short_term_dev_flow_{s_id}"] = flow - rolling_median

    # 2. Production-relative features
    total_prod = df_out[[f"production_M{i}" for i in range(1, 9)]].sum(axis=1) + 1.0
    for s_id in SENSORS:
        new_cols[f"flow_per_prod_{s_id}"] = df_out[f"flow_{s_id}"] / total_prod
        residual_col = f"ml_flow_residual_{s_id}"
        if residual_col in df_out.columns:
            new_cols[f"residual_per_prod_{s_id}"] = df_out[residual_col] / total_prod

    df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    # Lags leave NaNs at the head of the frame; back-fill so the first rows remain
    # scorable rather than being dropped.
    return df_out.bfill()
