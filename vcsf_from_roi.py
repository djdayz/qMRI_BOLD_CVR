import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes

csf_path = "csf2bold.nii"         # csf in bold space
roi_path = "vcsf_roi.nii"         # hand-drawn ventricle ROI
out_path = "vcsf_final.nii"

thr = 0.2
minvox = 20        # remove tiny specks

csf_img = nib.load(csf_path)
roi_img = nib.load(roi_path)

csf = csf_img.get_fdata()
roi = roi_img.get_fdata() > 0

# csf-like voxels, restricted to ROI
csf_bin = (csf > thr) & roi

# Fill small holes almost slice-wise; 3D fill
csf_bin = binary_fill_holes(csf_bin)

# Remove tiny components but keep all ventricular blobs in ROI
struct = generate_binary_structure(3, 2)  # 26-connectivity
lbl, n = label(csf_bin, structure=struct)

if n == 0:
    raise RuntimeError("No vCSF voxels found. Lower thr or check ROI alignment.")

counts = np.bincount(lbl.ravel())
counts[0] = 0

keep = np.zeros_like(csf_bin, dtype=np.uint8)
for lab in range(1, n + 1):
    if counts[lab] >= minvox:
        keep[lbl == lab] = 1

out = nib.Nifti1Image(keep, csf_img.affine, csf_img.header)
out.set_data_dtype(np.uint8)
nib.save(out, out_path)

print("Saved:", out_path, "| voxels:", int(keep.sum()), "| components kept:", int((counts[1:] >= minvox).sum()))
