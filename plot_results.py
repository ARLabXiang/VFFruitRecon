from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
run_type = "vit_swint"
pairable_df = pd.read_csv(f"{run_type}/real_tomato_matched_{run_type}_ms.csv")
mask_size_thresholds = np.linspace(100, 15000, 200)
sphere_thresholds = np.linspace(0.9, 4.0, 200)  # choose range & resolution
scatter_thres = 10000
scatter_E = 1.5
scatter_RANSAC_THRESH = 250     # not needed for when using the thresholds
error_calibrate_m = 1.53    # from synthetic data
error_calibrate_b = 1.41    # from synthetic data

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
plt.savefig(f"{run_type}/corr.pdf")


########
print("Plotting scatter")
fig, ax = plt.subplots(figsize=(5, 4.5))

s_handles, l_handles = [], []
def ransac_fit(x, y, residual_threshold=scatter_RANSAC_THRESH):
    X = x.reshape(-1, 1)

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=100,
        residual_threshold=residual_threshold,
        random_state=0,
        max_trials=1000
    )
    ransac.fit(X, y)

    inliers = ransac.inlier_mask_
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return slope, intercept, inliers

def plot_method(df, y_key, color, dark_color, label, s=12,marker='o',use_calibrate=True):
    mask = (df["masksize"] > scatter_thres) & (df["mesh_PCAS_0"] < scatter_E)
    df = df[mask]

    x = df["MeasuredSize"].values * 0.0254 * 100 # inch to cm
    y = df[y_key].values * 100
    if (use_calibrate):
        calibrated_y = (y-error_calibrate_b)/error_calibrate_m
    else:
        calibrated_y = y
    AE = abs(calibrated_y-x)
    APE = abs(calibrated_y-x)/x
    SE = AE**2
    print(f"{y_key} n: {len(df[y_key].values)}")
    print(f"{y_key} MAE: {AE.mean():.2f} cm")
    print(f"{y_key} MAPE: {APE.mean()*100:.2f} %")
    print(f"{y_key} RMSE: {np.sqrt(SE.mean()):.2f} cm")
    slope_r, intercept_r, inliers = ransac_fit(x, y)

    x_in = x[inliers]
    y_in = y[inliers]

    # Pearson r on all points
    r, _ = pearsonr(x, y)
    r_f, _ = pearsonr(x_in, y_in)
    print(f"{y_key} r: {r:.2f}, r_f: {r_f:.2f}")
    x_line = np.linspace(x.min(), x.max(), 200)

    s_handle = ax.scatter(
        x_in, y_in, s=s, alpha=0.3, color=color, marker=marker,
        label=f"{label} (n={len(x_in)}, $r$={r_f:.2f})"
    )
    ax.plot(
        x_line, slope_r * x_line + intercept_r,
        color="white", linewidth=3, alpha=0.6, zorder=4
    )

    l_ransac, = ax.plot(
        x_line, slope_r * x_line + intercept_r,
        color=dark_color, linewidth=1.5, zorder=5,
        label=f"{label} Fit (m={slope_r:.2f}, b={intercept_r:.2f})"
    )


    s_handles.extend([s_handle])
    l_handles.extend([l_ransac])
t_df = pd.read_csv(f"{run_type}/real_tomato_matched_{run_type}_ms.csv")
t_df["2D-LSeg"] *= 0.5/1729.537316070467 # 2D mask (metric) = 2D mask (pixel) * depth / focal length
t_df = t_df.rename(columns={
    "mesh_longest_diameter": "SAM3D",
    "RANSAC_PRadius": "RANSAC",
    "3D-LSeg": "LSeg3D",
    "2D-LSeg": "LSeg2D",
})
#plot_method(b_df, "SAM3D_SwinB", "red", "darkred", "Swin-B", s=15,marker='D')
plot_method(t_df, "SAM3D", "green", "darkgreen", f"{run_type}", s=15,marker='s',use_calibrate=True)
plot_method(t_df, "LSeg2D", "red", "darkred", f"{run_type}", s=15,marker='s',use_calibrate=False)

#plot_method(gt_df, "GT_diameter", "#69b3e7", "#1f77b4", "GT", s=15,marker='o')

handles = s_handles + l_handles
labels = [h.get_label() for h in handles]
ax.legend(handles, labels, loc="upper left", ncol=1,fontsize=10)

ax.set_xlabel("Measured Size (cm)",fontsize=14)
ax.set_ylabel("Estimated Size (cm)",fontsize=14)
ax.set_title(f"Greenhouse size estimate | Size > {scatter_thres}", x=0.45,fontsize=16)
ax.grid(True)
ax.tick_params(axis='both',labelsize=14)
ax.set_xlim(2.5, 6.8)
ax.set_ylim(5, 14.2)

plt.tight_layout()
plt.savefig(f"{run_type}/scatter.pdf")
#plt.show()
