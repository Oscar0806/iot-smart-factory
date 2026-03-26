import numpy as np
import pandas as pd
from datetime import datetime, timedelta
 
def generate_factory_data(n_machines=8, n_hours=24, sample_rate_min=5,
                           anomaly_machines=None):
    """
    Simulate IoT sensor data from a smart factory floor.
    Each machine has: temperature, vibration, pressure, humidity,
    power consumption sensors.
    
    anomaly_machines: list of machine IDs that will have anomalies
    """
    np.random.seed(42)
    if anomaly_machines is None:
        anomaly_machines = [3, 6]  # Machine 3 and 6 have issues
    
    n_samples = int(n_hours * 60 / sample_rate_min)
    start_time = datetime.now() - timedelta(hours=n_hours)
    timestamps = [start_time + timedelta(minutes=i*sample_rate_min)
                  for i in range(n_samples)]
    
    machine_types = {
        1: "CNC Lathe", 2: "Milling Machine", 3: "Hydraulic Press",
        4: "Robot Arm", 5: "Conveyor Belt", 6: "Injection Molder",
        7: "Welding Station", 8: "Assembly Robot",
    }
    
    all_data = []
    for m_id in range(1, n_machines + 1):
        m_type = machine_types.get(m_id, f"Machine {m_id}")
        has_anomaly = m_id in anomaly_machines
        
        for i, ts in enumerate(timestamps):
            # Time-of-day pattern (higher during shifts)
            hour = ts.hour
            shift_factor = 1.0 if 6 <= hour <= 22 else 0.3
            
            # Base sensor readings
            temp = 45 + 15 * shift_factor + np.random.normal(0, 2)
            vibration = 1.5 + 0.8 * shift_factor + np.random.normal(0, 0.2)
            pressure = 4.5 + 1.0 * shift_factor + np.random.normal(0, 0.3)
            humidity = 40 + 10 * shift_factor + np.random.normal(0, 3)
            power = 15 + 8 * shift_factor + np.random.normal(0, 1.5)
            
            # Inject anomalies
            is_anomaly = False
            if has_anomaly and m_id == 3:
                # Overheating: temperature gradually rises after hour 12
                hours_in = i * sample_rate_min / 60
                if hours_in > 12:
                    overheat = (hours_in - 12) ** 1.5 * 2
                    temp += overheat
                    if overheat > 15:
                        is_anomaly = True
            
            if has_anomaly and m_id == 6:
                # Excessive vibration: periodic spikes
                if i % 20 == 0 and i > n_samples * 0.4:
                    vibration += np.random.uniform(3, 6)
                    is_anomaly = True
            
            all_data.append({
                "timestamp": ts,
                "machine_id": m_id,
                "machine_type": m_type,
                "temperature_C": round(temp, 1),
                "vibration_mm_s": round(vibration, 2),
                "pressure_bar": round(pressure, 2),
                "humidity_pct": round(humidity, 1),
                "power_kW": round(power, 2),
                "is_anomaly": is_anomaly,
                "shift": "Day" if 6 <= hour <= 14 else
                         "Evening" if 14 < hour <= 22 else "Night",
            })
    
    return pd.DataFrame(all_data)
 
if __name__ == "__main__":
    df = generate_factory_data()
    df.to_csv("factory_data.csv", index=False)
    print(f"Generated {len(df)} sensor readings")
    print(f"Machines: {df['machine_id'].nunique()}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Anomalies: {df['is_anomaly'].sum()}")
    print("Saved to factory_data.csv")
