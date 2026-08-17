import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Global Conflict Early-Warning System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

/* ================================
   GLOBAL
   ================================ */

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}


/* ================================
   SIDEBAR
   ================================ */

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}


/* ================================
   TYPOGRAPHY
   ================================ */

h1 {
    font-size: 42px !important;
    font-weight: 750 !important;
    letter-spacing: -1.5px;
    color: #0f172a;
}

h2 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-top: 40px;
    color: #0f172a;
}

h3 {
    font-size: 20px !important;
    font-weight: 650 !important;
    color: #334155;
}


/* ================================
   METRIC CARDS
   ================================ */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 32px !important;
    font-weight: 750 !important;
}


/* ================================
   INFO BOX
   ================================ */

.info-box {
    padding: 22px 24px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}


/* ================================
   SECTION CONTAINERS
   ================================ */

div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: #ffffff;
}


/* ================================
   DATAFRAMES
   ================================ */

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}


/* ================================
   BUTTONS
   ================================ */

.stButton > button {
    border-radius: 10px;
    border: 1px solid #cbd5e1;
    padding: 8px 18px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #64748b;
}


/* ================================
   DIVIDERS
   ================================ */

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 35px 0;
}


/* ================================
   RISK LABELS
   ================================ */

.risk-high {
    color: #dc2626;
    font-weight: 700;
}

.risk-medium {
    color: #d97706;
    font-weight: 700;
}

.risk-low {
    color: #16a34a;
    font-weight: 700;
}


/* ================================
   HEADER
   ================================ */

.gci-header {
    padding: 10px 0 30px 0;
}

.gci-eyebrow {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.gci-title {
    font-size: 44px;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -2px;
    color: #0f172a;
}

.gci-subtitle {
    color: #64748b;
    font-size: 17px;
    margin-top: 10px;
    max-width: 800px;
}


/* ================================
   STATUS BADGE
   ================================ */

.gci-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.gci-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
}


/* ================================
   PLOTLY
   ================================ */

.js-plotly-plot {
    border-radius: 16px;
    overflow: hidden;
}


/* ================================
   FOOTER
   ================================ */

.gci-footer {
    margin-top: 60px;
    padding-top: 25px;
    border-top: 1px solid #e2e8f0;
    color: #94a3b8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    risk = pd.read_csv("global_risk_map_data.csv")

    calibration = pd.read_csv(
        "calibration_analysis.csv"
    )

    temporal = pd.read_csv(
        "temporal_stability_analysis.csv"
    )

    ablation = pd.read_csv(
        "ablation_model_comparison.csv"
    )

    threshold = pd.read_csv(
        "threshold_comparison.csv"
    )

    importance = pd.read_csv(
        "full_model_feature_importance.csv"
    )

    return (
        risk,
        calibration,
        temporal,
        ablation,
        threshold,
        importance
    )


try:

    (
        risk,
        calibration,
        temporal,
        ablation,
        threshold,
        importance
    ) = load_data()

except Exception as e:

    st.error(
        "Unable to load the project datasets."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# FIX DATA TYPES
# =========================================================

if "month" in risk.columns:
    risk["month"] = pd.to_datetime(
        risk["month"],
        errors="coerce"
    )


# =========================================================
# MODEL RESULTS
# =========================================================

ROC_GB = 0.882
PR_GB = 0.287
ROC_LR = 0.844
BRIER = 0.022
BASELINE_PR = 0.033
PR_LIFT = 8.73
TOTAL_EVENTS = 273


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌍 Global Conflict")

st.sidebar.caption(
    "Early-Warning & Risk Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Monitor",
        "Model Performance",
        "Early Warning",
        "Diagnostics",
        "Methodology"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning • Statistical Modelling • Geospatial Analytics"
)



# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    # =====================================================
    # HERO
    # =====================================================

    st.caption("GLOBAL CONFLICT EARLY-WARNING SYSTEM")

    st.title("Monitoring conflict escalation risk worldwide.")

    st.markdown(
        "A country-month machine-learning framework estimating "
        "the probability of major conflict escalation within "
        "the following six months."
    )


    st.divider()

    st.subheader("Model performance")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("ROC-AUC", f"{ROC_GB:.3f}")

    with c2:
        st.metric("PR-AUC", f"{PR_GB:.3f}")

    with c3:
        st.metric("Brier Score", f"{BRIER:.3f}")

    with c4:
        st.metric("PR-AUC Lift", f"{PR_LIFT:.2f}×")

    with c5:
        st.metric("Major Escalations", f"{TOTAL_EVENTS:,}")

    st.divider()

    # =====================================================
    # WHAT THE SYSTEM DOES
    # =====================================================

    st.markdown("## What does the system do?")

    st.markdown("**From historical conflict data to six-month escalation risk.**")

    st.write(
        "The system operates at the **country-month** level. "
        "For each observation, only information available at that "
        "point in time is used to estimate the probability of a "
        "major conflict escalation during the following six months."
    )

    st.write(
        "The result is a continuous probability rather than a binary "
        "prediction. This allows countries to be ranked by relative "
        "risk and different alert thresholds to be applied depending "
        "on the desired balance between missed events and false alarms."
    )

    # =====================================================
    # DATA + MODEL
    # =====================================================

    st.markdown("## Data & modelling")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-box">

            <div class="gci-eyebrow">
                DATA & FEATURE ENGINEERING
            </div>

            <h3>Historical conflict dynamics</h3>

            <p style="color:#64748b;">
            The model combines conflict history with country-level
            socioeconomic and political information.
            </p>

            <ul>
                <li>Lagged conflict indicators</li>
                <li>Rolling fatality measures</li>
                <li>Recent conflict dynamics</li>
                <li>Regime-change indicators</li>
                <li>GDP per capita</li>
                <li>External socioeconomic variables</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-box">

            <div class="gci-eyebrow">
                MACHINE LEARNING
            </div>

            <h3>Gradient Boosting Classifier</h3>

            <p style="color:#64748b;">
            The primary nonlinear model is compared against a
            simpler Logistic Regression benchmark.
            </p>

            <p style="color:#64748b;">
            The model produces continuous probability estimates
            that can be used for risk ranking and operational
            early-warning alerts.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # TEMPORAL VALIDATION
    # =====================================================

    st.markdown("## Out-of-sample validation")

    v1, v2 = st.columns([2, 1])

    with v1:

        st.markdown(
            """
            <div class="info-box">

            <div class="gci-eyebrow">
                CHRONOLOGICAL VALIDATION
            </div>

            <h3>Testing the model in the future, not the past</h3>

            <p style="color:#64748b;">
            The system uses a chronological train-test split.
            This reflects the intended prospective use of the
            model and prevents observations from later periods
            from influencing model development.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with v2:

        st.metric(
            "TRAINING",
            "1989–2019"
        )

        st.metric(
            "OUT-OF-SAMPLE",
            "2020–2025"
        )

    st.divider()

    # =====================================================
    # MODEL VALIDATION
    # =====================================================

    st.markdown("## Model validation")

    st.markdown(
        "Out-of-sample performance on the 2020–2025 validation period."
    )

    validation_col1, validation_col2 = st.columns(2)

    with validation_col1:

        st.markdown(
            """
            <div class="info-box">
                <div class="gci-eyebrow">PRIMARY MODEL</div>
                <h3 style="margin:0;">Gradient Boosting</h3>
                <p style="color:#64748b;">
                    Nonlinear model used for the main six-month
                    escalation risk estimates.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)

        with m1:
            st.metric("ROC-AUC", f"{ROC_GB:.3f}")

        with m2:
            st.metric("PR-AUC", f"{PR_GB:.3f}")


    with validation_col2:

        st.markdown(
            """
            <div class="info-box">
                <div class="gci-eyebrow">BENCHMARK</div>
                <h3 style="margin:0;">Logistic Regression</h3>
                <p style="color:#64748b;">
                    Simpler benchmark used to assess the incremental
                    value of the nonlinear model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("ROC-AUC", f"{ROC_LR:.3f}")

    st.divider()


    # =========================================================
    # OPERATIONAL EARLY WARNING
    # =========================================================

    st.markdown("## From prediction to early warning")

    st.markdown(
        "Continuous risk probabilities can be converted into "
        "different operational alert levels."
    )

    ew1, ew2 = st.columns(2)

    with ew1:

        st.markdown(
            """
            <div class="info-box">
                <div class="gci-eyebrow">CONSERVATIVE MODE</div>
                <h3 style="margin:0;">Fewer alerts</h3>
                <p style="color:#64748b;">
                    Designed to prioritize precision and limit
                    false alarms.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("Alert threshold", "0.50")


    with ew2:

        st.markdown(
            """
            <div class="info-box">
                <div class="gci-eyebrow">EARLY-WARNING MODE</div>
                <h3 style="margin:0;">Higher sensitivity</h3>
                <p style="color:#64748b;">
                    Designed to identify a larger share of potential
                    escalations at the cost of additional alerts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("Alert threshold", "0.20")

    st.divider()


    # =========================================================
    # EXPLORE THE SYSTEM
    # =========================================================

    st.markdown("## Explore the system")

    e1, e2, e3 = st.columns(3)

    with e1:

        st.markdown(
            """
            <div class="info-box">
                <div style="font-size:28px;">🌍</div>
                <h3>Risk Monitor</h3>
                <p style="color:#64748b;">
                    Explore country-level risk, warning alerts,
                    observed conflict intensity and scenario projections.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with e2:

        st.markdown(
            """
            <div class="info-box">
                <div style="font-size:28px;">📊</div>
                <h3>Model Performance</h3>
                <p style="color:#64748b;">
                    Examine discrimination, calibration,
                    thresholds and model comparisons.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with e3:

        st.markdown(
            """
            <div class="info-box">
                <div style="font-size:28px;">⚠️</div>
                <h3>Early Warning</h3>
                <p style="color:#64748b;">
                    Assess how effectively predicted risk
                    concentrates subsequent escalation events.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


    # =========================================================
    # INTERPRETATION
    # =========================================================

    st.markdown("## Interpreting the results")

    st.markdown(
        "**This is an analytical early-warning system, not an "
        "autonomous conflict prediction engine.**"
    )

    st.write(
        "Predicted probabilities represent statistical estimates "
        "based on historical relationships. A high-risk score does "
        "not mean that escalation will necessarily occur, while a "
        "low-risk score does not guarantee that escalation will not occur."
    )

    st.write(
        "Results should therefore be interpreted alongside geopolitical "
        "analysis, expert assessment and other relevant information."
    )

    # =========================================================
    # PROJECT SCOPE
    # =========================================================

    st.markdown("## Project scope")

    scope_data = pd.DataFrame({
        "Component": [
            "Unit of analysis",
            "Prediction horizon",
            "Training period",
            "Test period",
            "Primary model",
            "Benchmark",
            "Evaluation",
            "Application"
        ],
        "Description": [
            "Country-month",
            "Six months",
            "1989–2019",
            "2020–2025",
            "Gradient Boosting Classifier",
            "Logistic Regression",
            "ROC-AUC, PR-AUC, Brier Score, calibration, temporal stability and early-warning performance",
            "Conflict monitoring, risk ranking and early-warning analytics"
        ]
    })

    st.dataframe(
        scope_data,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================
    # FOOTER
    # =========================================================

    st.markdown(
        """
        <div class="gci-footer">
            Global Conflict Early-Warning System · Statistical research
            and decision-support framework · 1989–2025
        </div>
        """,
        unsafe_allow_html=True
    )
# =========================================================
# RISK MONITOR
# =========================================================

elif page == "Risk Monitor":

    st.title("Global Risk Monitor")

    st.markdown(
        "Monitor current conflict escalation risk, early-warning "
        "signals, scenario projections and observed conflict intensity."
    )

    st.divider()

    # =====================================================
    # CURRENT RISK SNAPSHOT
    # =====================================================

    current = risk.copy()

    latest_date = current["month"].max()

    current = current[
        current["month"] == latest_date
    ].copy()

    current["country_plot"] = (
        current["country"]
        .replace({
            "Yemen (North Yemen)": "Yemen",
            "Russia (Soviet Union)": "Russia",
            "Myanmar (Burma)": "Myanmar",
            "DR Congo (Zaire)":
                "Democratic Republic of the Congo"
        })
    )

    current["alert"] = (
        current["predicted_risk"] >= 0.20
    )

    total_countries = len(current)

    alert_count = int(
        current["alert"].sum()
    )

    mean_risk = current[
        "predicted_risk"
    ].mean()

    highest_risk = current[
        "predicted_risk"
    ].max()

    # =====================================================
    # SNAPSHOT METRICS
    # =====================================================

    st.subheader("Current risk snapshot")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Countries monitored",
        f"{total_countries:,}"
    )

    m2.metric(
        "Early-warning alerts",
        f"{alert_count:,}"
    )

    m3.metric(
        "Average predicted risk",
        f"{mean_risk:.1%}"
    )

    m4.metric(
        "Highest predicted risk",
        f"{highest_risk:.1%}"
    )

    st.caption(
        f"Latest available observation: "
        f"{latest_date.strftime('%B %Y')}"
    )

    st.divider()

    # =====================================================
    # MAP SELECTOR
    # =====================================================

    st.subheader("Risk maps")

    map_type = st.radio(
        "Select view",
        [
            "Global Conflict Risk",
            "6-Month Early-Warning Risk",
            "Five-Year Conflict Risk Projection",
            "Observed Conflict Intensity"
        ],
        horizontal=True
    )

    st.divider()

    # =====================================================
    # 1. GLOBAL CONFLICT RISK
    # =====================================================

    if map_type == "Global Conflict Risk":

        st.markdown(
            "### Global Conflict Risk"
        )

        st.write(
            "Predicted probability of major conflict escalation "
            "within the following six months."
        )

        fig = px.choropleth(
            current,
            locations="country_plot",
            locationmode="country names",
            color="predicted_risk",
            hover_name="country",
            color_continuous_scale="Reds",
            range_color=[0, 1],
            projection="natural earth",
            labels={
                "predicted_risk": "6-month risk"
            }
        )

        fig.update_layout(
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Higher values indicate higher estimated probability "
            "of major escalation within six months."
        )

    # =====================================================
    # 2. EARLY WARNING
    # =====================================================

    elif map_type == "6-Month Early-Warning Risk":

        st.markdown(
            "### 6-Month Early-Warning Risk"
        )

        st.write(
            "Countries with predicted risk at or above 0.20 "
            "are flagged for further analytical attention."
        )

        current["alert_status"] = np.where(
            current["predicted_risk"] >= 0.20,
            "Early-warning alert",
            "Below threshold"
        )

        fig = px.choropleth(
            current,
            locations="country_plot",
            locationmode="country names",
            color="alert_status",
            hover_name="country",
            color_discrete_map={
                "Early-warning alert": "#dc2626",
                "Below threshold": "#e5e7eb"
            },
            projection="natural earth"
        )

        fig.update_layout(
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            ),
            legend_title_text="Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        alerts = (
            current[
                current["predicted_risk"] >= 0.20
            ]
            .sort_values(
                "predicted_risk",
                ascending=False
            )
        )

        st.subheader(
            "Current early-warning alerts"
        )

        if len(alerts) > 0:

            alert_table = alerts[
                [
                    "country",
                    "predicted_risk"
                ]
            ].head(20).copy()

            alert_table["predicted_risk"] = (
                alert_table["predicted_risk"]
                .map(lambda x: f"{x:.1%}")
            )

            st.dataframe(
                alert_table,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No countries currently exceed the 0.20 "
                "early-warning threshold."
            )

    # =====================================================
    # 3. FIVE YEAR PROJECTION
    # =====================================================

    elif map_type == "Five-Year Conflict Risk Projection":

        st.markdown(
            "### Five-Year Conflict Risk Projection"
        )

        st.write(
            "Baseline scenario for 2026–2030 assuming that "
            "latest observed country-level conditions remain constant."
        )

        st.components.v1.html(
            open(
                "global_conflict_five_year_projection.html",
                encoding="utf-8"
            ).read(),
            height=700,
            scrolling=False
        )

        st.info(
            "Scenario analysis rather than a literal forecast. "
            "The projection assumes constant underlying conditions."
        )

    # =====================================================
    # 4. OBSERVED CONFLICT INTENSITY
    # =====================================================

    elif map_type == "Observed Conflict Intensity":

        st.markdown(
            "### Observed Conflict Intensity"
        )

        st.write(
            "Recorded conflict intensity measured using "
            "**fatalities per 100,000 inhabitants**."
        )

        observed = risk.copy()

        observed["intensity_log"] = np.log1p(
            observed["fatalities_per_100k"].clip(lower=0)
        )

        fig = px.choropleth(
            observed,
            locations="country",
            locationmode="country names",
            color="intensity_log",
            hover_name="country",
            color_continuous_scale="Reds",
            projection="natural earth",
            labels={
                "intensity_log": "Conflict intensity"
            },
            hover_data={
                "fatalities_per_100k": ":.2f"
            }
        )

        fig.update_layout(
            height=650,
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Colour intensity uses a logarithmic transformation "
            "for visualisation. Tooltip values show the original "
            "fatalities per 100,000 inhabitants."
        )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown(
        "Out-of-sample evaluation of the conflict escalation model. "
        "The model is trained on **1989–2019** and evaluated on "
        "previously unseen observations from **2020–2025**."
    )

    st.divider()

    # =====================================================
    # PERFORMANCE SNAPSHOT
    # =====================================================

    st.subheader("Performance snapshot")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "GB ROC-AUC",
        f"{ROC_GB:.3f}"
    )

    c2.metric(
        "GB PR-AUC",
        f"{PR_GB:.3f}"
    )

    c3.metric(
        "LR ROC-AUC",
        f"{ROC_LR:.3f}"
    )

    c4.metric(
        "PR-AUC Lift",
        f"{PR_LIFT:.2f}×"
    )

    c5.metric(
        "Brier Score",
        f"{BRIER:.3f}"
    )

    st.caption(
        "All metrics are calculated on the 2020–2025 out-of-sample period."
    )

    st.divider()

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader("Model comparison")

    st.write(
        "Gradient Boosting is compared with Logistic Regression as "
        "a simpler benchmark. Both models are evaluated on the same "
        "out-of-sample observations."
    )

    comparison = pd.DataFrame({
        "Model": [
            "Gradient Boosting",
            "Logistic Regression"
        ],
        "ROC-AUC": [
            ROC_GB,
            ROC_LR
        ]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="ROC-AUC",
        text="ROC-AUC"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        height=450,
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        f"The Gradient Boosting model achieves a ROC-AUC of "
        f"**{ROC_GB:.3f}**, compared with **{ROC_LR:.3f}** for "
        f"Logistic Regression."
    )

    st.divider()

    # =====================================================
    # PRECISION-RECALL
    # =====================================================

    st.subheader("Precision-recall performance")

    pr1, pr2 = st.columns([2, 1])

    with pr1:

        st.write(
            "Because major escalation events are relatively rare, "
            "PR-AUC provides an important complementary measure of "
            "performance."
        )

        st.write(
            f"The model achieves a PR-AUC of **{PR_GB:.3f}**, "
            f"against an event-prevalence baseline of "
            f"**{BASELINE_PR:.3f}**."
        )

    with pr2:

        st.metric(
            "PR-AUC",
            f"{PR_GB:.3f}"
        )

        st.metric(
            "Lift over baseline",
            f"{PR_LIFT:.2f}×"
        )

    st.divider()

    # =====================================================
    # CALIBRATION
    # =====================================================

    st.subheader("Probability calibration")

    st.write(
        "Calibration evaluates whether predicted probabilities "
        "correspond to the observed frequency of escalation events."
    )

    if len(calibration.columns) >= 2:

        x_col = calibration.columns[0]
        y_col = calibration.columns[1]

        fig = px.line(
            calibration,
            x=x_col,
            y=y_col,
            markers=True
        )

        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(
                dash="dash"
            )
        )

        fig.update_layout(
            height=500,
            xaxis_title="Predicted probability",
            yaxis_title="Observed frequency",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        f"Brier Score: {BRIER:.3f}. Lower values indicate better "
        "probabilistic accuracy."
    )

    st.divider()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("How to interpret these results")

    st.write(
        f"The Gradient Boosting model provides stronger out-of-sample "
        f"discrimination than the Logistic Regression benchmark "
        f"({ROC_GB:.3f} vs {ROC_LR:.3f} ROC-AUC)."
    )

    st.write(
        f"The PR-AUC is {PR_LIFT:.2f}× the event-prevalence baseline, "
        "indicating that the model concentrates a larger share of "
        "subsequent escalation events among higher-risk observations."
    )

    st.write(
        "These metrics evaluate statistical performance on previously "
        "unseen observations. They do not imply that individual "
        "escalation events can be predicted with certainty."
    )

# =========================================================
# EARLY WARNING
# =========================================================

elif page == "Early Warning":

    st.title("Early-Warning Analytics")

    st.markdown(
        "The model can be used to prioritise country-month observations "
        "according to their estimated probability of major conflict "
        "escalation within six months."
    )

    st.divider()

    # =====================================================
    # OPERATIONAL MODES
    # =====================================================

    st.subheader("Operational alert modes")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Early-Warning Mode",
            "0.20"
        )

        st.write(
            "Designed to favour recall. More observations are flagged "
            "in order to reduce the risk of missing potential escalations."
        )

    with c2:

        st.metric(
            "Conservative Mode",
            "0.50"
        )

        st.write(
            "More selective alerting with greater emphasis on precision "
            "and fewer observations requiring investigation."
        )

    st.caption(
        "Threshold selection reflects the operational trade-off between "
        "false alarms and missed escalation events."
    )

    st.divider()

    # =====================================================
    # THRESHOLD PERFORMANCE
    # =====================================================

    st.subheader("Threshold performance")

    if len(threshold) > 0:

        display_threshold = threshold.copy()

        metric_columns = [
            col for col in
            ["precision", "recall", "f1", "f1_score"]
            if col in display_threshold.columns
        ]

        if (
            "threshold" in display_threshold.columns
            and len(metric_columns) > 0
        ):

            fig = px.line(
                display_threshold,
                x="threshold",
                y=metric_columns,
                markers=True
            )

            fig.add_vline(
                x=0.20,
                line_dash="dash",
                annotation_text="Early-warning 0.20",
                annotation_position="top"
            )

            fig.add_vline(
                x=0.50,
                line_dash="dot",
                annotation_text="Conservative 0.50",
                annotation_position="bottom"
            )

            fig.update_layout(
                height=500,
                xaxis_title="Alert threshold",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.write(
            "Lower thresholds increase recall by flagging more "
            "observations. Higher thresholds produce fewer and more "
            "selective alerts."
        )

    st.divider()

    # =====================================================
    # PRECISION / RECALL
    # =====================================================

    st.subheader("Precision–recall trade-off")

    if (
        "threshold" in threshold.columns
        and "precision" in threshold.columns
        and "recall" in threshold.columns
    ):

        fig = px.line(
            threshold,
            x="threshold",
            y=["precision", "recall"],
            markers=True
        )

        fig.add_vline(
            x=0.20,
            line_dash="dash",
            annotation_text="0.20",
            annotation_position="top"
        )

        fig.add_vline(
            x=0.50,
            line_dash="dot",
            annotation_text="0.50",
            annotation_position="bottom"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Alert threshold",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.write(
        "A recall-oriented system accepts more false alarms in exchange "
        "for capturing a larger share of subsequent escalation events. "
        "A precision-oriented system reduces the number of alerts but "
        "risks missing more events."
    )

    st.divider()

    # =====================================================
    # RISK CONCENTRATION
    # =====================================================

    st.subheader("Risk concentration")

    st.write(
        "The ranking analysis measures how effectively the model "
        "concentrates observed escalation events within the highest-risk "
        "country-month observations."
    )

    ranking = pd.DataFrame({
        "Risk-ranked population": [
            "Top 1%",
            "Top 5%",
            "Top 10%",
            "Top 20%",
            "Top 30%"
        ],
        "Observations": [
            83,
            415,
            831,
            1663,
            2494
        ],
        "Observed events": [
            40,
            124,
            157,
            207,
            233
        ],
        "Precision": [
            0.482,
            0.299,
            0.189,
            0.124,
            0.093
        ],
        "Recall": [
            0.147,
            0.454,
            0.575,
            0.758,
            0.853
        ]
    })

    r1, r2 = st.columns([1, 2])

    with r1:

        st.dataframe(
            ranking.style.format({
                "Precision": "{:.1%}",
                "Recall": "{:.1%}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with r2:

        fig = px.bar(
            ranking,
            x="Risk-ranked population",
            y="Recall",
            text="Recall"
        )

        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
        )

        fig.update_layout(
            height=400,
            xaxis_title="Share of observations monitored",
            yaxis_title="Escalation events captured",
            yaxis_range=[0, 1],
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    st.markdown(
            """
            ### Key finding

            The ranking analysis measures how effectively the model
            concentrates observed escalation events within the highest-risk
            country-month observations.

            **Monitoring the top 5% of highest-risk country-month observations
            captures 45.4% of observed escalation events.** Expanding coverage
            to the top 10% captures 57.5%, while the top 20% captures 75.8%.
            """
        )
    st.divider()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("How should the alerts be interpreted?")

    st.info(
        "Risk scores are prioritisation signals, not deterministic "
        "predictions. A high-risk observation does not imply that an "
        "escalation will occur, while an observation below the selected "
        "threshold is not necessarily safe."
    )
# =========================================================
# DIAGNOSTICS
# =========================================================

elif page == "Diagnostics":

    st.title("Model Diagnostics")

    st.markdown(
        """
        A diagnostic assessment of model reliability, temporal
        stability, calibration, feature contribution and the
        incremental value of external socioeconomic and
        demographic information.
        """
    )

    st.divider()

    # =====================================================
    # 1. CALIBRATION
    # =====================================================

    st.header("Probability Calibration")

    st.markdown(
        """
        Calibration evaluates whether predicted probabilities
        correspond to the observed frequency of escalation events.

        A well-calibrated model should produce probabilities that
        approximately match the empirical event frequency.
        """
    )

    # -----------------------------------------------------
    # CALIBRATION KPIs
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Brier Score",
        f"{BRIER:.3f}",
        help="Lower values indicate better probabilistic accuracy."
    )

    c2.metric(
        "Baseline PR-AUC",
        f"{BASELINE_PR:.3f}",
        help="Observed prevalence of major escalation events."
    )

    c3.metric(
        "PR-AUC Lift",
        f"{PR_LIFT:.2f}×",
        help="Model PR-AUC relative to the prevalence baseline."
    )

    st.markdown("### Calibration Curve")

    # -----------------------------------------------------
    # DETECT CALIBRATION COLUMN NAMES
    # -----------------------------------------------------

    calibration_plot = calibration.copy()

    # Try to identify probability / observed columns
    probability_col = None
    observed_col = None

    for col in calibration_plot.columns:

        col_lower = col.lower()

        if (
            "predicted" in col_lower
            and "prob" in col_lower
        ):
            probability_col = col

        elif (
            "mean_predicted" in col_lower
            or "predicted_probability" in col_lower
        ):
            probability_col = col

        if (
            "observed" in col_lower
            and "frequency" in col_lower
        ):
            observed_col = col

        elif (
            "fraction" in col_lower
            and "positive" in col_lower
        ):
            observed_col = col

    # Fallback based on the known calibration structure
    if probability_col is None:
        possible = [
            c for c in calibration_plot.columns
            if "pred" in c.lower()
        ]

        if possible:
            probability_col = possible[0]

    if observed_col is None:
        possible = [
            c for c in calibration_plot.columns
            if (
                "observ" in c.lower()
                or "fraction" in c.lower()
            )
        ]

        if possible:
            observed_col = possible[0]

    if (
        probability_col is not None
        and observed_col is not None
    ):

        calibration_plot = calibration_plot[
            [
                probability_col,
                observed_col
            ]
        ].dropna()

        calibration_plot.columns = [
            "Predicted Probability",
            "Observed Frequency"
        ]

        # -------------------------------------------------
        # PERFECT CALIBRATION LINE
        # -------------------------------------------------

        fig_calibration = px.line(
            calibration_plot,
            x="Predicted Probability",
            y="Observed Frequency",
            markers=True,
            title="Predicted vs Observed Escalation Probability"
        )

        # Perfect calibration reference
        max_value = max(
            calibration_plot["Predicted Probability"].max(),
            calibration_plot["Observed Frequency"].max()
        )

        fig_calibration.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_value,
            y1=max_value,
            line=dict(
                dash="dash",
                color="gray"
            )
        )

        fig_calibration.update_layout(
            height=500,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Frequency"
        )

        st.plotly_chart(
            fig_calibration,
            use_container_width=True
        )

        st.caption(
            "The dashed line represents perfect calibration. "
            "Points close to the line indicate that predicted "
            "probabilities are broadly consistent with observed "
            "event frequencies."
        )

        # -------------------------------------------------
        # CALIBRATION TABLE
        # -------------------------------------------------

        st.markdown("### Calibration Table")

        st.dataframe(
            calibration_plot.style.format(
                {
                    "Predicted Probability": "{:.3f}",
                    "Observed Frequency": "{:.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "Predicted probabilities broadly follow the observed frequency "
            "of escalation events. Deviations from the diagonal indicate "
            "where the model tends to over- or under-estimate risk."
        )
    else:

        st.warning(
            "Calibration columns could not be identified automatically."
        )

    st.divider()

    # =====================================================
    # 2. TEMPORAL STABILITY
    # =====================================================

    st.header("Temporal Stability")

    st.markdown(
        """
        Model performance is evaluated separately across the
        out-of-sample period to determine whether predictive
        performance remains reasonably stable across different
        geopolitical periods.
        """
    )

    # -----------------------------------------------------
    # TEMPORAL KPIs
    # -----------------------------------------------------

    temporal_plot = temporal.copy()

    # Standardise column names where possible
    temporal_plot.columns = [
        str(c).strip()
        for c in temporal_plot.columns
    ]

    # Find columns
    year_col = None
    roc_col = None
    pr_col = None
    brier_col = None

    for col in temporal_plot.columns:

        c = col.lower()

        if c == "year":
            year_col = col

        if "roc" in c and "auc" in c:
            roc_col = col

        if "pr" in c and "auc" in c:
            pr_col = col

        if "brier" in c:
            brier_col = col

    if year_col is not None:

        temporal_plot[year_col] = pd.to_numeric(
            temporal_plot[year_col],
            errors="coerce"
        )

        temporal_plot = temporal_plot.sort_values(
            year_col
        )

    # -----------------------------------------------------
    # CALCULATE SUMMARY VALUES
    # -----------------------------------------------------

    if roc_col is not None:

        mean_roc = temporal_plot[roc_col].mean()
        min_roc = temporal_plot[roc_col].min()
        max_roc = temporal_plot[roc_col].max()

    else:

        mean_roc = np.nan
        min_roc = np.nan
        max_roc = np.nan

    if pr_col is not None:
        mean_pr = temporal_plot[pr_col].mean()
    else:
        mean_pr = np.nan

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean Yearly ROC-AUC",
        f"{mean_roc:.3f}"
        if not np.isnan(mean_roc)
        else "N/A"
    )

    c2.metric(
        "Minimum ROC-AUC",
        f"{min_roc:.3f}"
        if not np.isnan(min_roc)
        else "N/A"
    )

    c3.metric(
        "Maximum ROC-AUC",
        f"{max_roc:.3f}"
        if not np.isnan(max_roc)
        else "N/A"
    )

    c4.metric(
        "Mean Yearly PR-AUC",
        f"{mean_pr:.3f}"
        if not np.isnan(mean_pr)
        else "N/A"
    )

    # -----------------------------------------------------
    # ROC-AUC BY YEAR
    # -----------------------------------------------------

    if (
        year_col is not None
        and roc_col is not None
    ):

        st.markdown("### ROC-AUC by Year")

        fig_roc_year = px.line(
            temporal_plot,
            x=year_col,
            y=roc_col,
            markers=True,
            title="Out-of-Sample ROC-AUC by Year"
        )

        fig_roc_year.update_yaxes(
            range=[
                max(0, temporal_plot[roc_col].min() - 0.05),
                min(1, temporal_plot[roc_col].max() + 0.05)
            ]
        )

        fig_roc_year.update_layout(
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig_roc_year,
            use_container_width=True
        )

    # -----------------------------------------------------
    # PR-AUC BY YEAR
    # -----------------------------------------------------

    if (
        year_col is not None
        and pr_col is not None
    ):

        st.markdown("### PR-AUC by Year")

        fig_pr_year = px.line(
            temporal_plot,
            x=year_col,
            y=pr_col,
            markers=True,
            title="Out-of-Sample PR-AUC by Year"
        )

        fig_pr_year.update_layout(
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig_pr_year,
            use_container_width=True
        )

    # -----------------------------------------------------
    # YEARLY PERFORMANCE TABLE
    # -----------------------------------------------------

    st.markdown("### Yearly Validation Results")

    st.dataframe(
        temporal_plot,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Performance remains relatively stable across the 2020–2025 "
        "out-of-sample period, although individual years show some variation "
        "in predictive performance."
    )

    st.divider()

    # =====================================================
    # 3. ABLATION STUDY
    # =====================================================

    st.header("Incremental Value of External Features")

    st.markdown(
        """
        The ablation analysis tests whether socioeconomic and
        demographic variables provide predictive information beyond
        historical conflict characteristics.
        """
    )

    # -----------------------------------------------------
    # STANDARDISE ABLATION DATA
    # -----------------------------------------------------

    ablation_plot = ablation.copy()

    ablation_plot.columns = [
        str(c).strip()
        for c in ablation_plot.columns
    ]

    # Identify model column
    model_col = None

    for col in ablation_plot.columns:

        c = col.lower()

        if (
            "model" in c
            or "features" in c
        ):
            model_col = col
            break

    if model_col is None:

        # Usually the first non-numeric column
        object_columns = (
            ablation_plot
            .select_dtypes(
                include=["object"]
            )
            .columns
        )

        if len(object_columns) > 0:
            model_col = object_columns[0]

    # Identify metric columns
    roc_ablation_col = None
    pr_ablation_col = None
    brier_ablation_col = None

    for col in ablation_plot.columns:

        c = col.lower()

        if "roc" in c and "auc" in c:
            roc_ablation_col = col

        elif "pr" in c and "auc" in c:
            pr_ablation_col = col

        elif "brier" in c:
            brier_ablation_col = col

    # -----------------------------------------------------
    # DISPLAY ABLATION TABLE
    # -----------------------------------------------------

    st.markdown("### Model Specification Comparison")

    st.dataframe(
        ablation_plot,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The comparison shows the incremental value of adding socioeconomic "
        "and demographic information to historical conflict features. "
        "Differences in performance should be interpreted as predictive "
        "contribution, not causal effects."
    )

    # -----------------------------------------------------
    # ABLATION CHART
    # -----------------------------------------------------

    if (
        model_col is not None
        and roc_ablation_col is not None
        and pr_ablation_col is not None
    ):

        chart_data = ablation_plot[
            [
                model_col,
                roc_ablation_col,
                pr_ablation_col
            ]
        ].copy()

        chart_data = chart_data.melt(
            id_vars=[model_col],
            var_name="Metric",
            value_name="Score"
        )

        fig_ablation = px.bar(
            chart_data,
            x=model_col,
            y="Score",
            color="Metric",
            barmode="group",
            title="Predictive Performance by Model Specification"
        )

        fig_ablation.update_layout(
            height=500,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig_ablation,
            use_container_width=True
        )

    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    st.info(
        """
        The full model combines historical conflict dynamics with
        external socioeconomic and demographic variables.

        In the out-of-sample comparison, the external variables
        provide additional information primarily in terms of
        Precision-Recall performance. The results should be
        interpreted as incremental predictive value rather than
        evidence of causal relationships.
        """
    )

    st.divider()

    # =====================================================
    # 4. FEATURE IMPORTANCE
    # =====================================================

    st.header("What Drives the Predictions?")

    st.markdown(
        """
        Permutation importance measures how much predictive
        performance changes when individual features are randomly
        shuffled. Higher values indicate greater contribution to
        predictive discrimination.

        These values describe predictive association, not causal
        effects.
        """
    )

    importance_plot = importance.copy()

    importance_plot.columns = [
        str(c).strip()
        for c in importance_plot.columns
    ]

    # -----------------------------------------------------
    # IDENTIFY COLUMNS
    # -----------------------------------------------------

    feature_col = None
    importance_col = None
    std_col = None

    for col in importance_plot.columns:

        c = col.lower()

        if c == "feature":
            feature_col = col

        elif (
            "importance" in c
            and (
                "mean" in c
                or "value" in c
            )
        ):
            importance_col = col

        elif (
            "importance" in c
            and "std" in c
        ):
            std_col = col

    # Fallbacks
    if feature_col is None:

        object_columns = (
            importance_plot
            .select_dtypes(
                include=["object"]
            )
            .columns
        )

        if len(object_columns) > 0:
            feature_col = object_columns[0]

    if importance_col is None:

        possible = [
            c for c in importance_plot.columns
            if "importance" in c.lower()
        ]

        if possible:
            importance_col = possible[0]

    if (
        feature_col is not None
        and importance_col is not None
    ):

        importance_plot[importance_col] = pd.to_numeric(
            importance_plot[importance_col],
            errors="coerce"
        )

        importance_plot = (
            importance_plot
            .dropna(
                subset=[importance_col]
            )
            .sort_values(
                importance_col,
                ascending=False
            )
            .head(10)
        )

        importance_plot = importance_plot.sort_values(
            importance_col,
            ascending=True
        )

        fig_importance = px.bar(
            importance_plot,
            x=importance_col,
            y=feature_col,
            orientation="h",
            title="Top 10 Features by Permutation Importance"
        )

        fig_importance.update_layout(
            height=550,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            xaxis_title="Permutation Importance",
            yaxis_title=""
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

        # Full table
        st.markdown("### Feature Importance Details")

        display_columns = [
            c for c in [
                feature_col,
                importance_col,
                std_col
            ]
            if c is not None
        ]

        st.dataframe(
            importance_plot[
                display_columns
            ].sort_values(
                importance_col,
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The table shows the features with the largest permutation "
            "importance. Higher values indicate a greater contribution to "
            "predictive performance when the feature is available to the model."
        )

    else:

        st.warning(
            "Feature importance columns could not be identified."
        )

    st.divider()

    # =====================================================
    # 5. OPERATIONAL THRESHOLDS
    # =====================================================

    st.header("Operational Threshold Analysis")

    st.markdown(
        """
        The model produces a continuous risk probability. Operational
        deployment requires selecting a threshold that determines
        when an observation becomes an alert.

        Two operating points are highlighted: a conservative
        threshold of **0.50** and an early-warning threshold of
        **0.20**.
        """
    )

    threshold_plot = threshold.copy()

    threshold_plot.columns = [
        str(c).strip()
        for c in threshold_plot.columns
    ]

    st.markdown("### Threshold Comparison")

    st.dataframe(
        threshold_plot,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Lower thresholds generate more alerts and favour recall, while "
        "higher thresholds produce fewer and more selective alerts. The "
        "0.20 and 0.50 thresholds represent the two operating points used "
        "in this dashboard."
    )

    # -----------------------------------------------------
    # PRECISION / RECALL
    # -----------------------------------------------------

    precision_col = None
    recall_col = None
    threshold_col = None

    for col in threshold_plot.columns:

        c = col.lower()

        if "precision" in c:
            precision_col = col

        elif "recall" in c:
            recall_col = col

        elif (
            "threshold" in c
            or c == "cutoff"
        ):
            threshold_col = col

    if (
        precision_col is not None
        and recall_col is not None
    ):

        threshold_chart = threshold_plot[
            [
                c for c in [
                    threshold_col,
                    precision_col,
                    recall_col
                ]
                if c is not None
            ]
        ].copy()

        threshold_chart = threshold_chart.melt(
            id_vars=[
                threshold_col
            ] if threshold_col else None,
            var_name="Metric",
            value_name="Value"
        )

        fig_threshold = px.bar(
            threshold_chart,
            x="Metric",
            y="Value",
            color=(
                threshold_col
                if threshold_col
                else None
            ),
            barmode="group",
            title="Precision–Recall Trade-off"
        )

        fig_threshold.update_layout(
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig_threshold,
            use_container_width=True
        )

    st.info(
        """
        The 0.20 Early-Warning threshold prioritizes recall and
        identifies a larger share of subsequent escalations, while
        the 0.50 Conservative threshold produces substantially fewer
        alerts and prioritizes precision.

        The appropriate operating point depends on the relative
        cost of missed escalations versus false alarms.
        """
    )

    # =====================================================
    # FINAL DIAGNOSTIC SUMMARY
    # =====================================================

    st.divider()

    st.header("Diagnostic Summary")

    st.markdown(
        """
        ### What the diagnostics tell us

        **Strong discrimination.**  
        The Gradient Boosting model achieves an out-of-sample
        ROC-AUC of **0.882**, outperforming the Logistic Regression
        benchmark.

        **Useful performance under class imbalance.**  
        The PR-AUC of **0.287** represents an approximately
        **8.73× improvement over the prevalence baseline**.

        **Reasonable probabilistic performance.**  
        The Brier Score of **0.022** indicates relatively low
        probabilistic prediction error.

        **Temporal robustness.**  
        Year-by-year validation shows that predictive performance
        remains strong throughout the 2020–2025 out-of-sample
        period, although performance varies between individual
        years.

        **External features add information.**  
        The ablation analysis indicates that socioeconomic and
        demographic variables provide incremental predictive
        information when combined with historical conflict
        characteristics, particularly for Precision-Recall
        performance.

        **Operational trade-offs are explicit.**  
        The threshold analysis demonstrates the practical trade-off
        between identifying more potential escalations and limiting
        false alarms.
        """
    )

    st.caption(
        "Diagnostics are intended to assess model reliability and "
        "operational usefulness, not to establish causal relationships."
    )

# =========================================================
# METHODOLOGY
# =========================================================

elif page == "Methodology":

    st.title("Methodology")

    st.markdown(
        """
        ### Analytical Framework

        The Global Conflict Early-Warning System operates at the
        **country-month** level and estimates the probability that
        a major conflict escalation will occur during the following
        six months.
        """
    )

    st.divider()

    with st.expander("Prediction Target", expanded=True):

        st.write(
            """
            Each country-month observation is assigned a binary target
            indicating whether a major conflict escalation occurs within
            the subsequent six-month period.
            """
        )

    with st.expander("Temporal Validation", expanded=True):

        st.write(
            """
            The model uses a chronological train-test split.

            **Training:** 1989–2019

            **Out-of-sample testing:** 2020–2025

            This prevents future observations from entering the training
            sample and provides a more realistic evaluation of prospective
            predictive performance.
            """
        )

    with st.expander("Models"):

        st.write(
            """
            The primary model is a **Gradient Boosting Classifier**.

            Logistic Regression is used as a benchmark model to determine
            whether the nonlinear model provides additional predictive
            value.
            """
        )

    with st.expander("Evaluation Metrics"):

        st.write(
            """
            **ROC-AUC** measures discrimination between observations
            that subsequently experience escalation and those that do not.

            **PR-AUC** is particularly informative because major escalation
            events are relatively rare.

            **Brier Score** evaluates the accuracy of predicted probabilities.
            """
        )

    with st.expander("Operational Thresholds"):

        st.write(
            """
            Two operational modes are presented:

            **Conservative — 0.50**

            Prioritizes precision and generates fewer alerts.

            **Early Warning — 0.20**

            Prioritizes recall and identifies a larger proportion of
            potential escalations at the cost of additional false alarms.
            """
        )

    with st.expander("Five-Year Projection"):

        st.write(
            """
            The 2026–2030 projection holds the latest observed
            country-level characteristics constant and applies the
            estimated six-month risk repeatedly across ten periods.

            This is a **scenario analysis**, not a literal geopolitical
            forecast.
            """
        )

    with st.expander("Limitations"):

        st.write(
            """
            Historical relationships between conflict indicators and
            subsequent escalation may not remain stable.

            Future conflicts may also involve political, military,
            economic, technological or social conditions that are not
            represented in the historical training data.

            The system should therefore be interpreted as an
            **analytical early-warning and decision-support tool**,
            not as an autonomous conflict prediction system.
            """
        )