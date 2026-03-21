# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets, image, surface
from nilearn.image import smooth_img
from nilearn.plotting import plot_surf_stat_map

print("🚀 Dense Cortical Figure (NeurIPS-style)")

# =========================
# PATHS
# =========================
subj = "subj01"

streams_path = "../nsd/nsddata/ppdata/subj01/func1pt8mm/roi/streams.nii.gz"

early_path   = f"../mrifeat/{subj}/{subj}_early_betas_ave_te.npy"
ventral_path = f"../mrifeat/{subj}/{subj}_ventral_betas_ave_te.npy"

# Ridge
z_ridge = np.load(f"../decoded/{subj}/{subj}_early_scores_init_latent.npy")
c_ridge = np.load(f"../decoded/{subj}/{subj}_ventral_scores_c.npy")

# MLP
z_mlp = np.load(f"../decoded/{subj}/{subj}_early_scores_init_latent_mlp_d2_w2048_frac1.0.npy")
c_mlp = np.load(f"../decoded/{subj}/{subj}_ventral_scores_c_mlp_d6_w2048_frac1.0.npy")

EARLY_LABEL = 1
VENTRAL_LABEL = 5

# =========================
# LOAD DATA
# =========================
streams_img = nib.load(streams_path)
streams = streams_img.get_fdata()

X_early   = np.load(early_path)
X_ventral = np.load(ventral_path)

early_mask   = streams == EARLY_LABEL
ventral_mask = streams == VENTRAL_LABEL

# =========================
# ALIGNMENT
# =========================
def voxelwise_alignment(X, latent):
    X_c = X - X.mean(0, keepdims=True)
    L_c = latent - latent.mean(0, keepdims=True)

    numer = X_c.T @ L_c
    denom = (
        np.linalg.norm(X_c, axis=0, keepdims=True).T *
        np.linalg.norm(L_c, axis=0, keepdims=True)
    ) + 1e-10

    return np.mean(numer / denom, axis=1)

print("⏳ Computing alignment...")

early_ridge   = voxelwise_alignment(X_early, z_ridge)
early_mlp     = voxelwise_alignment(X_early, z_mlp)

ventral_ridge = voxelwise_alignment(X_ventral, c_ridge)
ventral_mlp   = voxelwise_alignment(X_ventral, c_mlp)

# =========================
# ABS DIFF
# =========================
early_diff   = np.abs(early_mlp) - np.abs(early_ridge)
ventral_diff = np.abs(ventral_mlp) - np.abs(ventral_ridge)

# normalize
def normalize(x):
    return x / (np.percentile(np.abs(x), 99) + 1e-8)

early_diff   = normalize(early_diff)
ventral_diff = normalize(ventral_diff)

# =========================
# BUILD VOLUMES
# =========================
def build_volume(values, mask):
    vol = np.zeros(streams.shape)
    coords = np.array(np.where(mask)).T
    for i,(x,y,z) in enumerate(coords):
        if i < len(values):
            vol[x,y,z] = values[i]
    return vol

vol_early   = build_volume(early_diff, early_mask)
vol_ventral = build_volume(ventral_diff, ventral_mask)

# =========================
# PROCESS
# =========================
bg = datasets.load_mni152_template(resolution=1)

def process(vol):
    return image.resample_to_img(
        smooth_img(nib.Nifti1Image(vol, streams_img.affine), fwhm=10),
        bg
    )

fs = datasets.fetch_surf_fsaverage("fsaverage")

def project(nii):
    return (
        surface.vol_to_surf(nii, fs.pial_left),
        surface.vol_to_surf(nii, fs.pial_right)
    )

tex_early   = project(process(vol_early))
tex_ventral = project(process(vol_ventral))

# =========================
# GLOBAL SCALE
# =========================
all_vals = np.concatenate([
    tex_early[0], tex_early[1],
    tex_ventral[0], tex_ventral[1]
])
vmax = np.percentile(np.abs(all_vals), 99)
threshold = vmax * 0.02

from matplotlib.gridspec import GridSpec

# =========================
# FIGURE — grouped LH | gap | RH
# =========================
fig = plt.figure(figsize=(16, 5))

# 5 cols: LH_lat, LH_med, [spacer], RH_lat, RH_med
gs = GridSpec(
    2, 5,
    width_ratios=[1, 1, 0.08, 1, 1],   # spacer: 0.25 → 0.08
    wspace=0.0,                          # wspace: 0.02 → 0.0
    hspace=0.05,
    left=0.02, right=0.88, top=0.88, bottom=0.05
)

# map view index → GridSpec column (skip col 2)
col_map = [0, 1, 3, 4]

views       = [("left","lateral"), ("right","lateral"), ("left","medial"), ("right","medial")]
view_labels = ["LH Lat", "RH Lat", "LH Med", "RH Med"]
titles      = ["Early (Diffusion)", "Ventral (CLIP)"]
textures    = [tex_early, tex_ventral]

for row, (tex_l, tex_r) in enumerate(textures):
    for vi, (hemi, view) in enumerate(views):

        ax = fig.add_subplot(gs[row, col_map[vi]], projection='3d')

        mesh   = fs.infl_left  if hemi == "left" else fs.infl_right
        tex    = tex_l         if hemi == "left" else tex_r
        bg_map = fs.sulc_left  if hemi == "left" else fs.sulc_right

        plot_surf_stat_map(
            mesh, tex,
            hemi=hemi, view=view, bg_map=bg_map,
            cmap="hot", vmax=vmax, vmin=0,
            threshold=threshold, colorbar=False, axes=ax
        )

        if row == 0:
            ax.set_title(view_labels[vi], fontsize=12, pad=2)

        if vi == 0:
            ax.text2D(-0.12, 0.5, titles[row], transform=ax.transAxes,
                      fontsize=12, weight='bold', va='center', rotation=90)

# # =========================
# # LH / RH set headers
# # =========================
# fig.text(0.25, 0.95, "Lateral", ha='center', fontsize=12, weight='bold')
# fig.text(0.70, 0.95, "Medial",  ha='center', fontsize=12, weight='bold')

# =========================
# Colorbar
# =========================
cax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="hot"),
    cax=cax
)
cbar.set_label("Δ Alignment (Nonlinear − Linear)", fontsize=12)

# plt.suptitle("Nonlinear Decoding Improves Alignment in Higher Visual Cortex",
#              fontsize=13, y=1.01)

plt.savefig("brain_dense_style.png", dpi=300, bbox_inches='tight', format='png')
plt.show()

print("✅ Saved: brain_dense_style.png")