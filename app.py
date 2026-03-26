import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from simulator import generate_factory_data
from detector import analyze_factory, THRESHOLDS
 
st.set_page_config(page_title="IoT Smart Factory",
                   page_icon="\U0001f3ed", layout="wide")
 
st.title("\U0001f3ed IoT Smart Factory Monitoring Dashboard")
st.markdown("**Real-time sensor monitoring with anomaly detection "
            "\u2013 Industry 4.0 concept**")
st.divider()
 
# ── SIDEBAR ──
st.sidebar.header("\u2699\uFE0F Settings")
n_hours = st.sidebar.slider("Monitoring window (hours)", 6, 48, 24)
anomaly_m = st.sidebar.multiselect(
    "Machines with anomalies",
    list(range(1, 9)), default=[3, 6],
    format_func=lambda x: f"M{x}")
sel_machine = st.sidebar.selectbox(
    "Focus machine",
    list(range(1, 9)),
    format_func=lambda x: f"Machine {x}")
 
# ── GENERATE + ANALYZE ──
@st.cache_data
def load(nh, am):
    df = generate_factory_data(n_hours=nh, anomaly_machines=am)
    return analyze_factory(df)
 
result, alerts = load(n_hours, tuple(anomaly_m))
 
# ── KPIs ──
c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("Machines", 8)
with c2: st.metric("Sensors", "5 per machine")
with c3: st.metric("Readings", f"{len(result):,}")
with c4:
    n_alert = len(alerts)
    st.metric("Alerts", n_alert,
              delta="Action!" if n_alert > 0 else "OK",
              delta_color="inverse" if n_alert > 0 else "normal")
with c5:
    n_ml = result["ml_anomaly"].sum()
    st.metric("ML Anomalies", int(n_ml))
 
st.divider()
 
# ── FACTORY FLOOR HEATMAP ──
st.subheader("\U0001f5fa\uFE0F Factory Floor – Machine Health")
# Latest reading per machine
latest = result.groupby("machine_id").last().reset_index()
health_data = []
for _, row in latest.iterrows():
    n_anom = result[(result["machine_id"]==row["machine_id"]) &
                     (result["combined_anomaly"])].shape[0]
    health = "Critical" if n_anom > 10 else "Warning" if n_anom > 0 else "OK"
    health_data.append({"Machine": f"M{int(row['machine_id'])}",
        "Type": row["machine_type"], "Health": health,
        "Temp": row["temperature_C"], "Vibration": row["vibration_mm_s"],
        "Anomalies": n_anom})
 
hdf = pd.DataFrame(health_data)
fig_floor = px.bar(hdf, x="Machine", y="Anomalies", color="Health",
    color_discrete_map={"OK":"#27AE60","Warning":"#F39C12","Critical":"#E74C3C"},
    hover_data=["Type","Temp","Vibration"],
    title="Anomaly Count per Machine")
fig_floor.update_layout(height=300, template="plotly_white")
st.plotly_chart(fig_floor, use_container_width=True)
 
# ── SENSOR TRENDS ──
st.subheader(f"\U0001f4c8 Machine {sel_machine} – Sensor Trends")
m_data = result[result["machine_id"] == sel_machine].copy()
 
sensors = [("temperature_C","Temperature (°C)","#E74C3C"),
           ("vibration_mm_s","Vibration (mm/s)","#3498DB"),
           ("pressure_bar","Pressure (bar)","#9B59B6"),
           ("power_kW","Power (kW)","#F39C12")]
 
cols = st.columns(2)
for i, (col_name, label, color) in enumerate(sensors):
    with cols[i % 2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m_data["timestamp"], y=m_data[col_name],
            mode="lines", name=label,
            line=dict(color=color, width=2)))
        lim = THRESHOLDS[col_name]
        fig.add_hline(y=lim["max"], line_dash="dash",
                      line_color="orange",
                      annotation_text=f"Warning: {lim['max']}")
        fig.add_hline(y=lim["critical"], line_dash="dash",
                      line_color="red",
                      annotation_text=f"Critical: {lim['critical']}")
        # Mark anomalies
        anom = m_data[m_data["combined_anomaly"]]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom[col_name],
                mode="markers", name="Anomaly",
                marker=dict(color="red", size=6, symbol="x")))
        fig.update_layout(height=250, template="plotly_white",
                          title=label, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
 
# ── ML ANOMALY SCORE ──
st.subheader("\U0001f916 ML Anomaly Detection (Isolation Forest)")
fig_ml = go.Figure()
for m in sorted(result["machine_id"].unique()):
    md = result[result["machine_id"]==m]
    fig_ml.add_trace(go.Scatter(
        x=md["timestamp"], y=md["anomaly_score"],
        mode="lines", name=f"M{m}", opacity=0.7))
fig_ml.add_hline(y=0, line_dash="dash", line_color="gray",
                 annotation_text="Decision boundary")
fig_ml.update_layout(height=350, template="plotly_white",
    xaxis_title="Time", yaxis_title="Anomaly Score",
    legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig_ml, use_container_width=True)
 
# ── ALERT LOG ──
st.subheader("\u26A0\uFE0F Alert Log")
if len(alerts) > 0:
    st.dataframe(alerts.sort_values("timestamp", ascending=False),
                 use_container_width=True, height=250)
else:
    st.success("No alerts! All machines operating within limits.")
 
st.divider()
st.caption("IoT Smart Factory Monitoring | Industry 4.0 Concept | "
           "Built by Oscar Vincent Dbritto" 
           )
