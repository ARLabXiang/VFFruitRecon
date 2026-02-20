from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
run_type = "vit_swint"
pairable_df = pd.read_csv(f"{run_type}/real_tomato_matched_{run_type}_ms.csv")
mask_size_thresholds = np.linspace(100, 15000, 200)
sphere_thresholds = np.linspace(0.9, 4.0, 200)  # choose range & resolution

plt.hist(pairable_df["masksize"], bins=100)
results = []
for mask_size_threshold in tqdm(mask_size_thresholds):
    for sphere_threshold in sphere_thresholds:
        sphere_mask = (
            (pairable_df["masksize"] > mask_size_threshold) &
            (pairable_df["mesh_PCAS_0"] < sphere_threshold)
        )
        key = ["RANSAC_PRadius","mesh_longest_diameter","3D-LSeg","2D-LSeg"]

        subset = pairable_df.loc[sphere_mask, key+["MeasuredSize"]]
        ransac_x = subset[key[0]].to_numpy()
        sam_x = subset[key[1]].to_numpy()
        lseg3d_x = subset[key[2]].to_numpy()
        lseg2d_x = subset[key[3]].to_numpy()
        y = subset["MeasuredSize"].to_numpy()
    # Need at least 2 points
        if len(subset) < 2 or np.std(sam_x) == 0 or np.std(y) == 0:
            ransac_r, ransac_p = np.nan, np.nan
            sam_r, sam_p = np.nan, np.nan
            lseg3d_r, lseg3d_p = np.nan, np.nan
            lseg2d_r, lseg2d_p = np.nan, np.nan
            ell_r, ell_p = np.nan, np.nan
        else:
            ransac_r, ransac_p = pearsonr(
                subset[key[0]],
                subset["MeasuredSize"]
            )
            sam_r, sam_p = pearsonr(
                subset[key[1]],
                subset["MeasuredSize"]
            )
            lseg3d_r, lseg3d_p = pearsonr(
                subset[key[2]],
                subset["MeasuredSize"]
            )
            lseg2d_r, lseg2d_p = pearsonr(
                subset[key[3]],
                subset["MeasuredSize"]
            )
        #print(thres, r, p, len(subset))
        results.append({
            "mask_thres": mask_size_threshold,
            "sphere_thres": sphere_threshold,
            "n_samples": len(subset),
            "sam_r": sam_r,
            "sam_p_value": sam_p,
            "ransac_r": ransac_r,
            "ransac_p_value": ransac_p,
            "lseg3d_r": lseg3d_r,
            "lseg3d_p": lseg3d_p,
            "lseg2d_r": lseg2d_r,
            "lseg2d_p": lseg2d_p,
        })
max_pair = (39,0,0)
corr_df = pd.DataFrame(results)
corr_df.to_csv(f"{run_type}/real12corr.csv", index=False)
corr_sub = corr_df.iloc[max_pair[0]::200]
print(corr_sub)
fig, ax1 = plt.subplots(figsize=(7,5))

# Correlation
ax1.plot(
    corr_sub["mask_thres"], corr_sub["ransac_r"], label="RANSAC R", color = "blue"
)
ax1.plot(
    corr_sub["mask_thres"], corr_sub["sam_r"], label="SAM R", color = "red"
)
ax1.plot(
    corr_sub["mask_thres"], corr_sub["lseg3d_r"], label="3D Mask R", color = "green"
)
ax1.plot(
    corr_sub["mask_thres"], corr_sub["lseg2d_r"], label="2D Mask R", color = "purple"
)
ax1.set_xlabel("Minimum MaskSize")
ax1.set_xlabel("MinMaskSize")

ax1.set_ylabel("Pearson correlation r")
ax1.legend()
ax1.grid(True)
# Sample count
ax2 = ax1.twinx()
ax2.plot(corr_sub["mask_thres"], corr_sub["n_samples"])
ax2.set_ylabel("Num Samples")
plt.title(f"Pearson r at Elongation threshold: {corr_sub['sphere_thres'].mean()}")
plt.show()