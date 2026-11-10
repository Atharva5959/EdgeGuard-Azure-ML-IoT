
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path("data/prediction_history.csv")

st.set_page_config(
    page_title="EdgeGuard | Machine Monitoring",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 EdgeGuard")
st.subheader("Real-Time IoT Machine Failure Monitoring")
st.caption("Simulated sensor data • Local ML inference • Automatic refresh")

@st.fragment(run_every=2)
def live_dashboard():
    if not DATA_PATH.exists():
        st.warning(
            "No sensor history found. Start the sensor simulator "
            "in another terminal first."
        )
        return

    try:
        df = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.info("Waiting for sensor readings...")
        return

    if df.empty:
        st.info("Waiting for sensor readings...")
        return

    required = [
        "timestamp", "temperature", "vibration", "pressure",
        "rpm", "predicted_failure", "failure_probability"
    ]

    if not set(required).issubset(df.columns):
        st.error("The history file does not have the expected columns.")
        return

    latest = df.iloc[-1]
    failure = int(latest["predicted_failure"]) == 1

    st.caption(
        f"Last reading: {latest['timestamp']} | "
        f"Total readings: {len(df)}"
    )

    st.divider()

    # Current machine condition
    if failure:
        st.error("🚨 MACHINE FAILURE ALERT — Check the simulated readings")
    else:
        st.success("✅ MACHINE CONDITION: NORMAL")

    # Latest sensor values
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Temperature", f"{latest['temperature']:.1f} °C")
    c2.metric("Vibration", f"{latest['vibration']:.2f} mm/s")
    c3.metric("Pressure", f"{latest['pressure']:.2f} bar")
    c4.metric("Rotational speed", f"{int(latest['rpm'])} RPM")

    st.divider()

    # Prediction details
    left, right = st.columns(2)

    with left:
        st.markdown("### Model prediction")
        st.metric(
            "Failure probability",
            f"{latest['failure_probability']:.1%}"
        )
        st.progress(
            max(0.0, min(1.0, float(latest["failure_probability"])))
        )

    with right:
        st.markdown("### Prediction summary")
        counts = df["predicted_failure"].value_counts()
        normal_count = int(counts.get(0, 0))
        failure_count = int(counts.get(1, 0))

        st.metric("Normal predictions", normal_count)
        st.metric("Failure alerts", failure_count)

    st.divider()

    # Recent sensor trends
    st.markdown("### Sensor trends")

    recent = df.tail(60).copy()
    recent["reading"] = range(1, len(recent) + 1)
    recent = recent.set_index("reading")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("**Temperature (°C)**")
        st.line_chart(recent[["temperature"]])

        st.markdown("**Vibration (mm/s)**")
        st.line_chart(recent[["vibration"]])

    with chart2:
        st.markdown("**Pressure (bar)**")
        st.line_chart(recent[["pressure"]])

        st.markdown("**Rotational speed (RPM)**")
        st.line_chart(recent[["rpm"]])

    # Recent predictions
    st.markdown("### Recent prediction history")

    history = df.tail(15).iloc[::-1].copy()
    history["predicted_condition"] = history[
        "predicted_failure"
    ].map({0: "Normal", 1: "Failure Alert"})

    st.dataframe(
        history[
            [
                "timestamp", "temperature", "vibration",
                "pressure", "rpm", "predicted_condition",
                "failure_probability"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Educational prototype using synthetic data. "
        "Predictions are not validated for real industrial machinery."
    )


live_dashboard()
