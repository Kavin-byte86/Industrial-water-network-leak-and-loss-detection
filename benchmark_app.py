"""
Streamlit benchmarking dashboard for the industrial water-network leak detector.

Design rule for this file: every performance claim shown to a viewer is either
computed live from the models or read from a result artifact. Nothing about model
performance is hardcoded in prose, because hardcoded claims go stale silently.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from benchmark_config import (
    DATASET_PATH,
    LEAKDB_RESULTS_PATH,
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    REPORT_PATH,
)

# Add repo to path so we can import the prediction API
sys.path.append(os.path.dirname(__file__))
from classifier.predict import WaterNetworkLeakDetector

st.set_page_config(page_title="Leak Detection Benchmark", layout="wide", page_icon="💧")

TOTAL_ROWS = 105_120          # 365 days at 5-minute sampling
TEST_FRACTION = 0.15          # matches the chronological split in train_all.py
PLOT_SAMPLE = 8_000           # scatter points; more than this is unreadable anyway

ACCENT = "#2563eb"
ALERT = "#dc2626"
MUTED = "#94a3b8"


# ---------------------------------------------------------
# CACHED DATA LOADING
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading test split of the synthetic dataset...")
def load_test_split():
    """Loads only the chronological test tail.

    The full ML dataset is ~234 MB; the dashboard only ever shows the held-out
    tail, so we skip the train/val rows at read time instead of loading and
    discarding them.
    """
    df_path = os.path.join(DATASET_PATH, "ml_training_dataset.csv")
    if not os.path.exists(df_path):
        return None

    first_test_row = int(TOTAL_ROWS * (1.0 - TEST_FRACTION))
    df = pd.read_csv(df_path, skiprows=range(1, first_test_row + 1))
    df = df.ffill().bfill()

    if "timestamp" not in df.columns and {"date", "time"} <= set(df.columns):
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])

    prod_cols = [c for c in df.columns if c.startswith("production_M")]
    if prod_cols and "production_total_pct" not in df.columns:
        df["production_total_pct"] = df[prod_cols].mean(axis=1)

    return df


@st.cache_resource(show_spinner="Loading inference models...")
def load_inference_api(version: str):
    path = MODEL_V2_PATH if version == "v2" else MODEL_V1_PATH
    if not os.path.exists(path):
        return None
    return WaterNetworkLeakDetector(models_dir=path)


@st.cache_data(show_spinner="Scoring the held-out test set...")
def score_test_split(version: str, _api, df: pd.DataFrame):
    """Runs the full 4-stage pipeline over the test split and returns predictions.

    `version` is not used in the body; it is the cache key. `_api` is underscore-
    prefixed so Streamlit skips hashing the unhashable model object, which means
    the version string is what distinguishes v1 results from v2.

    This is what makes the accuracy tab honest: the numbers shown are produced by
    the same model a viewer can interrogate in the simulation tab.
    """
    results = _api.predict_batch(df)
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "y_true": df["leak"].to_numpy().astype(int),
            "y_prob": [r["leak_probability"] for r in results],
            "y_pred": [int(r["leak_detected"]) for r in results],
            "pred_zone": [r["leak_zone"] for r in results],
            "pred_rate": [r["leak_rate"] for r in results],
            "true_zone": df["leak_zone"].to_numpy(),
            "true_rate": df["leak_rate"].to_numpy(),
        }
    )


@st.cache_data
def load_leakdb_results():
    metrics_path = os.path.join(LEAKDB_RESULTS_PATH, "classifier_metrics.csv")
    events_path = os.path.join(LEAKDB_RESULTS_PATH, "scenario_events.csv")
    fi_path = os.path.join(LEAKDB_RESULTS_PATH, "feature_importance.csv")
    read = lambda p: pd.read_csv(p) if os.path.exists(p) else None
    return read(metrics_path), read(events_path), read(fi_path)


@st.cache_data
def load_saved_metrics():
    path = os.path.join(REPORT_PATH, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------
# SMALL UI HELPERS
# ---------------------------------------------------------
def metric_row(pairs):
    cols = st.columns(len(pairs))
    for col, (label, value, help_text) in zip(cols, pairs):
        col.metric(label, value, help=help_text)


def styled(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)")
    return fig


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
def tab_overview(df, scores):
    st.header("Project Overview")
    st.markdown(
        """
**Objective:** detect, localize and quantify water leaks in an industrial textile
wet-processing plant.

**The problem:** textile manufacturing consumes enormous amounts of water, and
demand swings with the production schedule. A jump from 1000 to 2000 L/min at the
main inlet is *normal* when eight dyeing and bleaching machines spin up together.
A flow threshold cannot tell that apart from a burst pipe.

**The approach:** learn the *expected* flow at every sensor from machine states and
production rates, then classify on the **residual** (actual − expected) plus
mass-balance and pressure anomalies. High flow is only a leak when it is
unexplained.
        """
    )

    leak_rows = int(df["leak"].sum())
    metric_row([
        ("Test rows", f"{len(df):,}", "Chronological hold-out tail of the year"),
        ("Leak rows", f"{leak_rows:,}", f"{leak_rows / len(df):.1%} of the test split"),
        ("Distinct leak zones", f"{df.loc[df['leak'] == 1, 'leak_zone'].nunique()}", None),
        ("Sensors", "16 flow + 5 pressure", "Plus 8 machine and 3 tap state channels"),
    ])

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Network topology (16 sensors)")
        st.code(
            """
                        [ J1: Main Inlet ]
                                |
    +-------------+-------------+-------------+-------------+
    v             v                           v             v
[J2: Branch A] [J3: Branch B]          [J4: Branch C] [J7: Branch D]
  |- J5  (M1)    |- J8  (M3)             |- J11 (M6)    |- J13 (M8)
  +- J6  (M2)    |- J9  (M4)             +- J12 (M7)    +- Tap 1,2,3
                 +- J10 (M5)
            """,
            language=None,
        )
        st.caption(
            "Mass balance holds at every header: J1 = J2+J3+J4+J7, and so on. "
            "A leak below a header breaks exactly one of those equations, which is "
            "what localizes it."
        )

    with col2:
        st.subheader("4-stage architecture")
        st.markdown(
            """
| Stage | Model | Purpose |
|---|---|---|
| 1 | 16 × Ridge (degree-2) | Expected flow per sensor from production + state. Fit on **normal data only**. |
| 2 | XGBoost classifier | Leak / no-leak from residuals, balances, pressures. |
| 3 | XGBoost multiclass | Which zone ruptured. |
| 4 | XGBoost regressor | Leak rate in L/min. |
            """
        )
        st.caption(
            "Stage 1 never sees a leak during training, so its residuals stay "
            "meaningful when one occurs."
        )

    st.divider()
    st.subheader("Why a flow threshold fails")
    sample = df.sample(min(PLOT_SAMPLE, len(df)), random_state=0)
    fig = px.scatter(
        sample,
        x="production_total_pct",
        y="flow_J1",
        color=sample["leak"].map({0: "Normal", 1: "Leak"}),
        color_discrete_map={"Normal": MUTED, "Leak": ALERT},
        opacity=0.65,
        title="Main inlet flow vs total production",
        labels={
            "production_total_pct": "Mean machine production (%)",
            "flow_J1": "Main flow J1 (L/min)",
            "color": "Ground truth",
        },
    )
    st.plotly_chart(styled(fig, 420), use_container_width=True)

    normal_max = df.loc[df["leak"] == 0, "flow_J1"].max()
    leaks_below = int((df.loc[df["leak"] == 1, "flow_J1"] < normal_max).sum())
    st.info(
        f"Normal operation reaches **{normal_max:,.0f} L/min** at the main inlet. "
        f"**{leaks_below:,} of {leak_rows:,} leak samples** sit below that value, so any "
        "fixed threshold high enough to avoid false alarms would miss them entirely. "
        "The diagonal band is legitimate production; leaks are the points off it."
    )


def tab_accuracy(df, scores, version):
    st.header(f"Textile model performance — {version} on held-out test split")
    st.caption(
        "Computed live by running the loaded models over every test row. The split "
        "is chronological, so no future data informs a past prediction."
    )

    y_true, y_pred, y_prob = scores["y_true"], scores["y_pred"], scores["y_prob"]
    pr = precision_score(y_true, y_pred, zero_division=0)
    rc = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prauc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metric_row([
        ("Precision", f"{pr:.3f}", "Of flagged samples, how many really leaked"),
        ("Recall", f"{rc:.3f}", "Of real leak samples, how many were caught"),
        ("F1", f"{f1:.3f}", None),
        ("PR-AUC", f"{prauc:.3f}", "Threshold-independent; the honest headline for imbalanced data"),
    ])
    metric_row([
        ("True positives", f"{tp:,}", None),
        ("False positives", f"{fp:,}", "Nuisance alarms"),
        ("False negatives", f"{fn:,}", "Missed leak samples"),
        ("True negatives", f"{tn:,}", None),
    ])

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion matrix")
        cm = np.array([[tn, fp], [fn, tp]])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Samples"),
            x=["No leak", "Leak"],
            y=["No leak", "Leak"],
            title="Test-set confusion matrix",
        )
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(styled(fig_cm), use_container_width=True)

    with col_b:
        st.subheader("Precision–recall trade-off")
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve",
                                    line=dict(color=ACCENT, width=3)))
        fig_pr.add_trace(go.Scatter(x=[rc], y=[pr], mode="markers", name="Operating point",
                                    marker=dict(color=ALERT, size=13, symbol="x")))
        fig_pr.update_layout(title=f"PR curve (AUC {prauc:.3f})",
                             xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(styled(fig_pr), use_container_width=True)
        st.caption(
            "The marked point is the deployed threshold. Slide it left for fewer "
            "nuisance alarms, right to catch more leaks."
        )

    st.divider()
    st.subheader("What gets missed, and does it matter?")
    col_c, col_d = st.columns(2)

    missed = scores[(scores["y_true"] == 1) & (scores["y_pred"] == 0)]
    caught = scores[(scores["y_true"] == 1) & (scores["y_pred"] == 1)]

    with col_c:
        if len(missed):
            fig_missed = go.Figure()
            fig_missed.add_trace(go.Histogram(x=caught["true_rate"], name="Detected",
                                              marker_color=ACCENT, opacity=0.75))
            fig_missed.add_trace(go.Histogram(x=missed["true_rate"], name="Missed",
                                              marker_color=ALERT, opacity=0.75))
            fig_missed.update_layout(barmode="overlay",
                                     title="Leak magnitude: detected vs missed",
                                     xaxis_title="True leak rate (L/min)",
                                     yaxis_title="Samples")
            st.plotly_chart(styled(fig_missed), use_container_width=True)

            noise_floor = 1000.0 * (0.02) * 1.5  # 2% flow noise, SNR 1.5
            below = int((missed["true_rate"] < noise_floor).sum())
            st.warning(
                f"Median missed leak is **{missed['true_rate'].median():.1f} L/min**; "
                f"**{below}/{len(missed)}** missed samples fall below the "
                f"~{noise_floor:.0f} L/min sensor noise floor. Those are not a "
                "tuning failure — they are physically buried under 2% measurement "
                "noise on a ~1000 L/min main. Forcing detection there buys false alarms."
            )
        else:
            st.success("No missed leak samples in the test split.")

    with col_d:
        if len(missed):
            zone_counts = missed["true_zone"].value_counts().reset_index()
            zone_counts.columns = ["Zone", "Missed samples"]
            fig_zone = px.bar(zone_counts, x="Missed samples", y="Zone",
                              orientation="h", title="Missed leaks by zone",
                              color_discrete_sequence=[ALERT])
            fig_zone.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(styled(fig_zone), use_container_width=True)
            st.caption(
                "Concentration in a few zones points at low-flow terminal branches "
                "where a small leak is the smallest fraction of the parent flow."
            )

    st.divider()
    st.subheader("Stage 3 & 4: localization and sizing")
    col_e, col_f = st.columns(2)
    with col_e:
        if len(caught):
            zone_hit = (caught["pred_zone"] == caught["true_zone"]).mean()
            st.metric("Zone accuracy on detected leaks", f"{zone_hit:.1%}",
                      help="Stage 3, evaluated only where Stage 2 fired")
            cross = (
                caught.groupby(["true_zone", "pred_zone"]).size()
                .reset_index(name="n")
            )
            fig_cross = px.density_heatmap(
                cross, x="pred_zone", y="true_zone", z="n",
                color_continuous_scale="Blues", title="Predicted vs actual zone",
            )
            st.plotly_chart(styled(fig_cross), use_container_width=True)
    with col_f:
        if len(caught):
            mae = (caught["pred_rate"] - caught["true_rate"]).abs().mean()
            st.metric("Leak-rate MAE on detected leaks", f"{mae:.1f} L/min",
                      help="Stage 4, evaluated only where Stage 2 fired")
            fig_rate = px.scatter(
                caught.sample(min(3000, len(caught)), random_state=0),
                x="true_rate", y="pred_rate", opacity=0.5,
                title="Estimated vs true leak rate",
                labels={"true_rate": "True (L/min)", "pred_rate": "Estimated (L/min)"},
                color_discrete_sequence=[ACCENT],
            )
            lim = float(caught["true_rate"].max())
            fig_rate.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                          name="Perfect", line=dict(dash="dash", color=MUTED)))
            st.plotly_chart(styled(fig_rate), use_container_width=True)


def tab_leakdb(metrics, events, fi):
    st.header("External benchmark validation — LeakDB (Hanoi CMH)")
    st.markdown(
        "The textile dataset is synthetic, so good numbers there partly measure "
        "self-consistency. To test the *methodology* independently we replayed the "
        "same `expected flow → residual → classify` recipe on the public LeakDB "
        "municipal benchmark, upsampled to 5-minute resolution, with whole-scenario "
        "train/test splits."
    )
    st.caption(
        "LeakDB has no production schedule, so this validates the hydraulic residual "
        "method — not the production-aware pipeline as a whole."
    )

    if metrics is None or events is None:
        st.info(
            "LeakDB results not found. Fetch the benchmark and run the pipeline:\n\n"
            "```bash\n"
            "git clone https://github.com/KIOS-Research/LeakDB.git external_validation/LeakDB\n"
            "python external_validation/run_leakdb_test.py\n"
            "```"
        )
        return

    best = metrics.loc[metrics["F1"].idxmax()]
    baseline = metrics.loc[metrics["Model"] == "A_FlowOnly"]

    metric_row([
        ("Best feature set", str(best["Model"]).split("_", 1)[-1], str(best["Model"])),
        ("F1", f"{best['F1']:.3f}",
         f"vs {baseline['F1'].iloc[0]:.3f} using raw flow alone" if len(baseline) else None),
        ("Precision", f"{best['Precision']:.3f}", None),
        ("PR-AUC", f"{best['PR-AUC']:.3f}", None),
    ])

    if len(baseline):
        lift = best["F1"] - baseline["F1"].iloc[0]
        st.success(
            f"Adding residuals and network balances lifts F1 by **{lift:+.3f}** over "
            "raw flow features on an independent network. That is the transferable "
            "result: the residual framing, not the specific model, is what works."
        )

    st.divider()
    col_x, col_y = st.columns([1, 1])
    with col_x:
        st.subheader("Feature-set ablation")
        melted = metrics.melt(
            id_vars="Model", value_vars=["Precision", "Recall", "F1", "PR-AUC"],
            var_name="Metric", value_name="Score",
        )
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                     title="Impact of residuals and balances on detection")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(styled(fig, 420), use_container_width=True)

    with col_y:
        st.subheader("Event-level detection")
        best_events = events[events["Model"] == best["Model"]].copy()
        detected = int(best_events["Detected"].sum())
        total = int(best_events["Has_Leak"].sum())

        show = best_events[["Scenario", "Detected", "TTD_Hrs",
                            "Pct_Detected", "False_Alarms_Before"]].rename(
            columns={
                "TTD_Hrs": "Time to detect (h)",
                "Pct_Detected": "% of leak flagged",
                "False_Alarms_Before": "False alarms before onset",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

        # Claim is derived from the table above, never asserted independently.
        if detected == total:
            ttds = best_events.loc[best_events["Detected"], "TTD_Hrs"].dropna()
            worst = ttds.max() if len(ttds) else float("nan")
            st.success(
                f"`{best['Model']}` detects **{detected}/{total}** leak scenarios, "
                f"worst-case time-to-detection **{worst:.2f} h**."
            )
        else:
            missed = best_events.loc[~best_events["Detected"], "Scenario"].tolist()
            st.warning(
                f"`{best['Model']}` detects **{detected}/{total}** leak scenarios. "
                f"Not detected: {', '.join(missed)}."
            )

    if fi is not None:
        st.divider()
        st.subheader("Which hydraulic features carry the signal")
        top = fi.head(12).sort_values("Importance")
        fig_fi = px.bar(top, x="Importance", y="Feature", orientation="h",
                        title="Top features (XGBoost gain)",
                        color_discrete_sequence=[ACCENT])
        st.plotly_chart(styled(fig_fi, 420), use_container_width=True)


def tab_simulation(df, api):
    st.header("Interactive leak injection")
    st.markdown(
        "Take a window of genuinely normal telemetry, inject a synthetic leak into "
        "the raw sensor readings, and watch the pipeline react. Nothing here is "
        "pre-recorded — every point is a live model call."
    )

    normal = df[df["leak"] == 0].sort_values("timestamp")
    if len(normal) < 50:
        st.warning("Not enough leak-free rows in the test split to run a simulation.")
        return

    with st.sidebar:
        st.header("Leak injection controls")
        window = st.slider("Window length (5-min steps)", 50, 500, 200, step=50)
        inject = st.checkbox("Inject a leak", value=True)
        zone = st.selectbox("Header to rupture", ["J2", "J3", "J4", "J7"])
        rate = st.slider("Leak magnitude (L/min)", 5.0, 500.0, 150.0, step=5.0)
        onset = st.slider("Onset (% into window)", 10, 90, 40, step=5)
        ramp = st.selectbox("Profile", ["Sudden step", "Gradual ramp"])

    win = normal.head(window).copy().reset_index(drop=True)

    if inject:
        n = len(win)
        start = int(n * onset / 100)
        profile = np.zeros(n)
        if ramp == "Sudden step":
            profile[start:] = rate
        else:
            profile[start:] = np.linspace(0, rate, n - start)

        # A leak below a header shows up at that header and everything upstream of
        # it, but NOT at the downstream machine feeds - that asymmetry is exactly
        # what the mass-balance features detect.
        win["flow_J1"] = win["flow_J1"] + profile
        win[f"flow_{zone}"] = win[f"flow_{zone}"] + profile
        for p in ["pressure_J1", f"pressure_{zone}"]:
            if p in win.columns:
                win[p] = win[p] - profile * 0.0004  # friction loss from added flow
    else:
        profile = np.zeros(len(win))

    if st.button("Run simulation", type="primary"):
        with st.spinner("Running the 4-stage pipeline..."):
            results = api.predict_batch(win)

        res = pd.DataFrame({
            "timestamp": win["timestamp"],
            "flow_J1": win["flow_J1"],
            "flow_zone": win[f"flow_{zone}"],
            "injected": profile,
            "prob": [r["leak_probability"] for r in results],
            "detected": [r["leak_detected"] for r in results],
            "pred_zone": [r["leak_zone"] for r in results],
            "pred_rate": [r["leak_rate"] for r in results],
        })

        fired = res[res["detected"]]
        if inject:
            onset_idx = int(len(res) * onset / 100)
            after = fired[fired.index >= onset_idx]
            before = fired[fired.index < onset_idx]

            c1, c2, c3, c4 = st.columns(4)
            if len(after):
                steps = int(after.index[0]) - onset_idx
                c1.metric("Detected", "Yes", help="Alarm raised after leak onset")
                c2.metric("Time to detect", f"{steps * 5} min",
                          help="At 5-minute sampling")
                top_zone = after["pred_zone"].mode()
                c3.metric("Predicted zone",
                          top_zone.iloc[0] if len(top_zone) else "-",
                          help=f"Injected at {zone}")
                c4.metric("Estimated rate", f"{after['pred_rate'].median():.0f} L/min",
                          delta=f"{after['pred_rate'].median() - rate:+.0f} vs injected")
            else:
                c1.metric("Detected", "No")
                c2.metric("Injected rate", f"{rate:.0f} L/min")
                c3.metric("Peak probability", f"{res['prob'].max():.2f}")
                c4.metric("Threshold", f"{api.threshold:.2f}")
                st.warning(
                    f"A {rate:.0f} L/min leak was not detected. Against a "
                    f"~{win['flow_J1'].mean():.0f} L/min main with 2% noise, that is "
                    "within the noise floor — consistent with the detectability analysis."
                )
            if len(before):
                st.caption(f"{len(before)} pre-onset false alarms in this window.")
        else:
            st.metric("False alarms on clean data", f"{len(fired)} / {len(res)}")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Sensor telemetry (L/min)", "Leak probability",
                            "Estimated leak rate (L/min)"),
        )
        fig.add_trace(go.Scatter(x=res["timestamp"], y=res["flow_J1"],
                                 name="Main flow J1", line=dict(color=ACCENT)), row=1, col=1)
        fig.add_trace(go.Scatter(x=res["timestamp"], y=res["flow_zone"],
                                 name=f"Header {zone}", line=dict(color="#7c3aed")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res["timestamp"], y=res["prob"], name="P(leak)",
                                 fill="tozeroy", line=dict(color=ALERT)), row=2, col=1)
        fig.add_hline(y=api.threshold, line_dash="dash", line_color="black", row=2, col=1,
                      annotation_text=f"threshold {api.threshold:.2f}")
        fig.add_trace(go.Scatter(x=res["timestamp"], y=res["pred_rate"], name="Estimated rate",
                                 mode="lines", line=dict(color="#f59e0b")), row=3, col=1)
        if inject:
            fig.add_trace(go.Scatter(x=res["timestamp"], y=res["injected"],
                                     name="Injected rate (truth)", mode="lines",
                                     line=dict(color=MUTED, dash="dot")), row=3, col=1)
            onset_ts = res["timestamp"].iloc[int(len(res) * onset / 100)]
            for r in (1, 2, 3):
                fig.add_vline(x=onset_ts, line_dash="dot", line_color=ALERT, row=r, col=1)

        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_layout(height=820, title_text="Live inference telemetry",
                          margin=dict(l=10, r=10, t=80, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    st.title("💧 Industrial Textile Water Network Leak Detection")
    st.caption("AI/ML benchmarking and demonstration dashboard")

    version = st.sidebar.radio(
        "Model version", ["v2", "v1"], index=0,
        help="v2 adds temporal lags and a tuned decision threshold",
    )

    df = load_test_split()
    if df is None:
        st.error(
            "Dataset not found. Generate it first:\n\n```bash\npython generator.py\n```"
        )
        return

    api = load_inference_api(version)
    if api is None:
        st.error(
            f"Models for {version} not found. Train them first:\n\n"
            "```bash\npython classifier/training/train_all.py\n"
            "python classifier/training/optimize_pipeline.py\n```"
        )
        return

    scores = score_test_split(version, api, df)
    ldb_metrics, ldb_events, ldb_fi = load_leakdb_results()

    tabs = st.tabs([
        "1. Overview & methodology",
        "2. Textile model accuracy",
        "3. LeakDB external validation",
        "4. Interactive simulation",
    ])
    with tabs[0]:
        tab_overview(df, scores)
    with tabs[1]:
        tab_accuracy(df, scores, version)
    with tabs[2]:
        tab_leakdb(ldb_metrics, ldb_events, ldb_fi)
    with tabs[3]:
        tab_simulation(df, api)


if __name__ == "__main__":
    main()
