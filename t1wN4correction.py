import os
import ants
import numpy as np

raw_t1 = "sub-01_ses-01_T1w.nii"
output_dir = "N4corrected_T1w"
os.makedirs(output_dir, exist_ok= True)

n4_out = os.path.join(output_dir, "T1w_N4_shr3.nii")
bias_out = os.path.join(output_dir, "T1w_bias_field_shr3.nii")
mask_out = os.path.join(output_dir, "T1w_N4_mask3.nii")

if not os.path.exists(raw_t1):
    raise FileNotFoundError(f"Input file not found: {raw_t1}")

print("Loading raw T1w image:", raw_t1)
t1 = ants.image_read(raw_t1)

mask = ants.get_mask(t1)
ants.image_write(mask, mask_out)

t1_n4 = ants.n4_bias_field_correction(t1, mask=mask,
                                      rescale_intensities=True,
                                      shrink_factor=3,
                                      convergence={"iters": [50, 50, 30, 20], "tol":1e-7},
                                      return_bias_field=False)

bias_est = t1/(t1_n4 + 1e-12)
ants.image_write(t1_n4, n4_out)
ants.image_write(bias_est, bias_out)
print("N4-corrected image: ", n4_out)
print("Estimated bias field: ", bias_out)

t1_data = t1.numpy()
t1n4_data = t1_n4.numpy()
m = mask.numpy().astype(bool)

mean_bef = np.mean(t1_data[m])
std_bef = np.std(t1_data[m])
mean_aft = np.mean(t1n4_data[m])
std_aft = np.std(t1n4_data[m])
cv_bef = std_bef/mean_bef
cv_aft = std_aft/mean_aft

print(f"mean bef:{mean_bef:.2f}, std bef: {std_bef:.2f}, CV before: {cv_bef:.3f}" )
print(f"mean aft:{mean_aft:.2f}, std aft: {std_aft:.2f}, CV after: {cv_aft:.3f}")
print("N4 correction completed")