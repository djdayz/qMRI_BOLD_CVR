import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

bold_path = "/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/sub-01/func/sub-01_ses-01_bold_mcf.nii"
etco2_path = "/Users/mac/PycharmProjects/pythonMPhysproject/EtCO2_results_baseline_mmHg.txt"

TR = 1.55

bold_img = nib.load(bold_path)
bold = bold_img.get_fdata()
nx, ny, nz, nt = bold.shape
V = nx * ny * nz
t_bold = np.arange(nt) * TR

# Reshape to (V, T)
bold_flat = bold.reshape(V, nt)

# LOAD ETCO2 file
etdf = pd.read_csv(etco2_path, sep=None, engine="python")
time_et = etdf["sec"].values

co2_cols = [c for c in etdf.columns if "etco2_interp" in c.lower()]
co2_col = co2_cols[0]
et_raw = etdf[co2_col].values

# Resample ETCO2 to BOLD grid
et_resampled = np.interp(t_bold, time_et, et_raw)



# Utility: zscore
def zscore(x):
    s = np.std(x)
    if s < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


# initial voxel
initial_voxel = 0
voxel_ts = bold_flat[initial_voxel, :]


# Build figure
plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.30)

initial_shift = 0
et_shifted = np.interp(t_bold - initial_shift, time_et, et_raw)

[line_bold] = ax.plot(t_bold, zscore(voxel_ts), linewidth=2, label=f"BOLD voxel (x = {nx}, y = {ny}, z = {nz})")
[line_co2] = ax.plot(t_bold, zscore(et_shifted),
                     linewidth=2, alpha=0.8,
                     label=f"EtCO₂ (shift = {initial_shift:.1f}s)")
ax.set_ylim(-3, 3)


ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized amplitude")
ax.set_title(f"BOLD vs EtCO₂ — Voxel index {initial_voxel}")
ax.grid()
ax.legend()


# voxel index slider
ax_vox = plt.axes([0.15, 0.17, 0.7, 0.03], facecolor="lightgray")
slider_vox = Slider(
    ax=ax_vox,
    label="Voxel index",
    valmin=0,
    valmax=V - 1,
    valinit=initial_voxel,
    valstep=1
)

# EtCO₂ shift slider
ax_shift = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor="lightgray")
slider_shift = Slider(
    ax=ax_shift,
    label="EtCO₂ shift (s)",
    valmin=-700,
    valmax=50,
    valinit=0,
    valstep=0.5
)

# Update function
def update(_):
    voxel_idx = int(slider_vox.val)
    shift = slider_shift.val

    # Get voxel signal
    voxel_ts = bold_flat[voxel_idx, :]

    # Shift EtCO₂
    et_shifted = np.interp(t_bold - shift, time_et, et_raw)

    # Update plots
    line_bold.set_ydata(zscore(voxel_ts))
    line_co2.set_ydata(zscore(et_shifted))

    # Update title + legend
    ax.set_title(f"BOLD vs EtCO₂ — Voxel index {voxel_idx}")
    line_co2.set_label(f"EtCO₂ (shift = {shift:.1f}s)")
    ax.legend()

    fig.canvas.draw_idle()


slider_vox.on_changed(update)
slider_shift.on_changed(update)

plt.show(block=True)
