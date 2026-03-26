import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
 
THRESHOLDS = {
    "temperature_C": {"min": 30, "max": 75, "critical": 85},
    "vibration_mm_s": {"min": 0.5, "max": 4.0, "critical": 6.0},
    "pressure_bar": {"min": 2.0, "max": 7.0, "critical": 8.5},
    "humidity_pct": {"min": 20, "max": 70, "critical": 80},
    "power_kW": {"min": 5, "max": 30, "critical": 35},
}
 
def check_thresholds(row):
    """Check sensor values against operational limits."""
    alerts = []
    for sensor, limits in THRESHOLDS.items():
        val = row[sensor]
        if val >= limits["critical"]:
            alerts.append(f"CRITICAL: {sensor}={val} (limit {limits['critical']})")
        elif val >= limits["max"]:
            alerts.append(f"WARNING: {sensor}={val} (limit {limits['max']})")
        elif val <= limits["min"]:
            alerts.append(f"LOW: {sensor}={val} (min {limits['min']})")
    return alerts
 
def run_ml_detection(df, contamination=0.05):
    """Isolation Forest anomaly detection on sensor data."""
    features = ["temperature_C", "vibration_mm_s", "pressure_bar",
                "humidity_pct", "power_kW"]
    X = df[features].values
    model = IsolationForest(contamination=contamination,
                             random_state=42, n_estimators=100)
    model.fit(X)
    scores = model.decision_function(X)
    predictions = model.predict(X)
    return scores, predictions == -1  # True = anomaly
 
def analyze_factory(df):
    """Full analysis: thresholds + ML."""
    # Threshold alerts
    all_alerts = []
    for _, row in df.iterrows():
        alerts = check_thresholds(row)
        for a in alerts:
            all_alerts.append({
                "timestamp": row["timestamp"],
                "machine_id": row["machine_id"],
                "machine_type": row["machine_type"],
                "alert": a,
            })
    
    # ML detection
    scores, ml_anomalies = run_ml_detection(df)
    df = df.copy()
    df["anomaly_score"] = scores
    df["ml_anomaly"] = ml_anomalies
    df["combined_anomaly"] = df["is_anomaly"] | df["ml_anomaly"]
    
    return df, pd.DataFrame(all_alerts)
 
if __name__ == "__main__":
    from simulator import generate_factory_data
    df = generate_factory_data()
    result, alerts = analyze_factory(df)
    print(f"Threshold alerts: {len(alerts)}")
    print(f"ML anomalies: {result['ml_anomaly'].sum()}")
    for m in result["machine_id"].unique():
        m_data = result[result["machine_id"] == m]
        n_anom = m_data["combined_anomaly"].sum()
        if n_anom > 0:
            print(f"  Machine {m}: {n_anom} anomalies")
