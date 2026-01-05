import os
import numpy as np
import nibabel as nib
from scipy.ndimage import generate_binary_structure, label, binary_erosion

cvr_mag_path = "/Users/mac/PycharmProjects/pythonMPhysproject/cvr_tcnr_check/CVR_mag_clipped.nii"    # Unit: %BOLD/mmHg
delay_path = "/Users/mac/PycharmProjects/pythonMPhysproject/cvr_tcnr_check/CVR_delay_clipped.nii"   # Unit: seconds
tcnr_path = "/Users/mac/PycharmProjects/pythonMPhysproject/cvr_tcnr_check/tCNR_pos.nii"
brain_mask_path = "boldmask3D_ero.nii"
csf_mask_path = "vcsf_final.nii"    # 3D VCSF mask in BOLD space
bold_4d_path = "/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir/sub-01/func/sub-01_ses-01_bold_mcf.nii"

outdir = "vessel_seg_multi"
os.makedirs(outdir, exist_ok=True)

def load_4d_float(path):
    img = nib.load(path)
    data = img.get_fdata().astype(float)
    if data.ndim != 4:
        raise ValueError(f"{path} must be 4D got {data.shape}")
    return data, img

def compute_tsnr(bold_4d, eps = 1e-6):
    mean_s = np.mean(bold_4d, axis=-1)
    std_s = np.std(bold_4d, axis=-1)
    return mean_s / (std_s + eps)

# Quality gate
tsnr_p = 10

# Keep voxels where CVR is detectable
tcnr_min = 2.5

# Vessel by CVR magnitude
cvr_percentile = 99    # top 1 %

# Delay constraint
delay_max_sec = 80.0    # keep physiological upper bound

# Cleanup
min_cluster_size = 60  # remove tiny speckles
cluster_connectivity = 2    # 2 gives 26-neighbour connectivity in 3D

out_name = "vessel_mask_multi.nii"

def load_3d_bool(path, target_shape=None):
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    mask = data > 0
    if target_shape is not None and mask.shape != target_shape:
        raise ValueError(f"{path} shape {mask.shape} does not match target shape {target_shape}")
    return mask, img

def load_3d_float(path, target_shape=None):
    img = nib.load(path)
    data = img.get_fdata().astype(float)
    if data.ndim != 3:
        raise ValueError(f"{path} must be 3D, got {data.shape}")
    if target_shape is not None and data.shape != target_shape:
        raise ValueError(f"{path} shape {data.shape} does not match target shape {target_shape}")
    return data, img

def size_filter(binary_mask, min_size, connectivity=2):
    """Remove connected components smaller than min_size."""
    if min_size is None or min_size <= 1:
        return binary_mask
    struct = generate_binary_structure(3, connectivity)
    lbl, n = label(binary_mask, structure=struct)
    if n == 0:
        return binary_mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes >= int(min_size)
    return keep[lbl]


cvr_mag, ref_img = load_3d_float(cvr_mag_path)
delay, _ = load_3d_float(delay_path, target_shape=cvr_mag.shape)
tcnr, _ = load_3d_float(tcnr_path, target_shape=cvr_mag.shape)
bold4d, _ = load_4d_float(bold_4d_path)
bold4d = bold4d[..., 30:]

tsnr = compute_tsnr(bold4d)

brain_mask, _ = load_3d_bool(brain_mask_path, target_shape=cvr_mag.shape)
brain_mask = binary_erosion(brain_mask, iterations=1)

tsnr_vals = tsnr[brain_mask & np.isfinite(tsnr)]
tsnr_thr = np.percentile(tsnr_vals, tsnr_p)

if csf_mask_path is not None:
    csf_mask, _ = load_3d_bool(csf_mask_path, target_shape=cvr_mag.shape)
else:
    csf_mask = np.zeros_like(brain_mask, dtype=bool)

finite = np.isfinite(cvr_mag) & np.isfinite(delay) & np.isfinite(tcnr)

# Quality gate
snr_good = brain_mask & np.isfinite(tsnr) & (tsnr >= tsnr_thr)

# SNR info
print(f"tSNR gate: p{tsnr_p}: tSNR_thr = {tsnr_thr:.2f}, voxels = {int(snr_good.sum())}")
good = snr_good & finite & (tcnr >= tcnr_min)

# CVR magnitude threshold; percentile within "good" voxels
cvr_vals = cvr_mag[good]
if cvr_vals.size == 0:
    raise RuntimeError("No voxels passed the tCNR gate; lower tcnr_min or check masks.")

cvr_thr = np.percentile(cvr_vals, cvr_percentile)
vessel = good & (cvr_mag >= cvr_thr)

print("CVR (good) min/median/max:", np.min(cvr_vals), np.median(cvr_vals), np.max(cvr_vals))
print("Delay (good) min/median/max:", np.nanmin(delay[good]), np.nanmedian(delay[good]), np.nanmax(delay[good]))
print("CSF voxels:", int(csf_mask.sum()))
print("Good and CSF:", int((good & csf_mask).sum()))

tmp = good & (cvr_mag >= cvr_thr)
print("After percentile only:", int(tmp.sum()))

tmp2 = tmp & (~csf_mask)
print("After CSF removal:", int(tmp2.sum()))

tmp3 = tmp2 & (delay <= float(delay_max_sec))
print("After delay_max:", int(tmp3.sum()))

# CSF subtraction
vessel &= (~csf_mask)

print("brain mask voxels: ", int(brain_mask.sum()))
print("tCNR gate voxels:", int(good.sum()))
print(f"CVR percentile threshold: {cvr_percentile} -> CVR_thr={cvr_thr:.4f}")
print("Voxels after CVR+delay+CSF:", int(vessel.sum()))

# Cleanup

vessel = size_filter(vessel, min_cluster_size, connectivity=cluster_connectivity)
print("after size-filter:", int(vessel.sum()))

#Save tSNR maps
nib.save(nib.Nifti1Image(tsnr.astype(np.float32), ref_img.affine, ref_img.header), os.path.join(outdir, "tSNR.nii"))
nib.save(nib.Nifti1Image(snr_good.astype(np.uint8), ref_img.affine, ref_img.header), os.path.join(outdir, "SNR_good_mask.nii"))

#Save vessel mask
out_path = os.path.join(outdir, out_name)
out_img = nib.Nifti1Image(vessel.astype(np.uint8), ref_img.affine, ref_img.header)
nib.save(out_img, out_path)

print("Saved vessel mask:", out_path)
print("Vessel voxels:", int(vessel.sum()))
