import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

# Load gas traces data
df = pd.read_csv("sub-01_ses-01_gas_traces.txt", sep='\t')

# Extract time and CO₂
time = df["sec"]
co2_pct = df["PctCO2"]

pct_to_mmhg = 760.0 / 100.0
co2 = co2_pct * pct_to_mmhg


# Find EtCO₂ peaks
peaks, _ = find_peaks(co2, distance=60, prominence=0.005)
etco2_times = time.iloc[peaks]
etco2_values = co2.iloc[peaks]

# Linear interpolation between detected EtCO₂ peaks
etco2_interp = np.interp(time, etco2_times, etco2_values)

etco2_baseline = np.median(etco2_values)
baseline_line = np.ones_like(time) * etco2_baseline

print(f"EtCO₂ baseline = {etco2_baseline:.2f} mmHg")

# Add interpolated EtCO₂ column to DataFrame
df["EtCO2_interp_mmHg"] = etco2_interp
df["EtCO2_baseline_mmHg"] = etco2_baseline

# === 7. Save to a new text file ===
output_path = "EtCO2_results_baseline_mmHg.txt"
df.to_csv(output_path, sep='\t', index=False, float_format='%.5f')

print(f" File saved: {output_path}")
print(f"Detected {len(etco2_values)} EtCO₂ peaks. Mean EtCO₂ = {etco2_values.mean():.3f}%")

# Plot for confirmation
plt.figure(figsize=(12, 6))
plt.plot(time, co2, color='blue', linewidth=1, label='PctCO₂ (%)')
plt.plot(etco2_times, etco2_values, color = "red", linewidth=1, label='Detected EtCO₂')
plt.plot(time, etco2_interp, 'g--', linewidth=1, label='Interpolated EtCO₂')
plt.plot(time, baseline_line, color='magenta', linewidth=2, label='EtCO₂ baseline (slow drift)')

plt.xlabel("Time (sec)")
plt.ylabel("CO₂ concentration (%)")
plt.title("Interpolated End-tidal CO₂ (EtCO₂) Trend")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
