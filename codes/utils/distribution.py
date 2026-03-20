import argparse
import os
from typing import List, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import rbf_kernel


def load_folder(path: str) -> Tuple[np.ndarray, List[str]]:
	"""
	Load either:
	1) a folder of per-sample .npy files, or
	2) a single .npy matrix with shape (N, D).

	Returns matrix (N, D) and an alignment key list.
	"""
	if os.path.isdir(path):
		files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])

		if not files:
			raise ValueError(f"No .npy files found in: {path}")

		data = []
		for name in files:
			arr = np.load(os.path.join(path, name))
			data.append(arr.reshape(-1))

		stacked = np.stack(data, axis=0).astype(np.float64)
		return stacked, files

	if os.path.isfile(path):
		arr = np.load(path)
		if arr.ndim == 1:
			arr = arr.reshape(-1, 1)
		elif arr.ndim > 2:
			arr = arr.reshape(arr.shape[0], -1)
		else:
			arr = arr.reshape(arr.shape[0], -1)

		stacked = arr.astype(np.float64)
		keys = [f"{i:06d}" for i in range(stacked.shape[0])]
		return stacked, keys

	raise ValueError(f"Path does not exist: {path}")

def assert_same_filenames(reference: List[str], other: List[str], tag: str) -> None:
	"""Ensure two sorted filename lists match exactly for sample alignment."""
	if reference == other:
		return

	n = min(len(reference), len(other))
	mismatch_idx = next((i for i in range(n) if reference[i] != other[i]), None)

	if mismatch_idx is None:
		raise ValueError(
			f"Filename mismatch for {tag}: different counts "
			f"({len(reference)} vs {len(other)})."
		)

	raise ValueError(
		f"Filename mismatch for {tag} at index {mismatch_idx}: "
		f"{reference[mismatch_idx]} != {other[mismatch_idx]}"
	)


def _diag_kl(mu_x: np.ndarray, var_x: np.ndarray, mu_y: np.ndarray, var_y: np.ndarray) -> float:
	"""KL divergence between diagonal Gaussians N_x || N_y."""
	diff2 = (mu_y - mu_x) ** 2
	ratio = var_x / var_y
	log_term = np.log(var_y) - np.log(var_x)
	return 0.5 * np.sum(log_term + ratio + diff2 / var_y - 1.0)


def _full_kl(mu_x: np.ndarray, cov_x: np.ndarray, mu_y: np.ndarray, cov_y: np.ndarray) -> float:
	"""KL divergence between full Gaussians N_x || N_y."""
	d = mu_x.shape[0]
	diff = mu_y - mu_x

	sign_x, logdet_x = np.linalg.slogdet(cov_x)
	sign_y, logdet_y = np.linalg.slogdet(cov_y)

	if sign_x <= 0 or sign_y <= 0:
		raise np.linalg.LinAlgError("Covariance is not positive definite after regularization.")

	solve_cov = np.linalg.solve(cov_y, cov_x)
	term1 = np.trace(solve_cov)
	term2 = float(diff.T @ np.linalg.solve(cov_y, diff))
	term3 = logdet_y - logdet_x
	return 0.5 * (term1 + term2 - d + term3)


def kl_gaussian(
	x: np.ndarray,
	y: np.ndarray,
	eps: float = 1e-6,
	mode: str = "auto",
	full_max_dim: int = 1024,
) -> float:
	"""
	KL divergence between fitted Gaussians N(x) || N(y).

	mode:
	  - "full": full covariance closed-form KL
	  - "diag": diagonal covariance approximation
	  - "auto": full if dimension <= full_max_dim, else diagonal
	"""
	if x.shape[1] != y.shape[1]:
		raise ValueError("KL requires x and y to have the same feature dimension.")

	mu_x = x.mean(axis=0)
	mu_y = y.mean(axis=0)

	if mode not in {"auto", "full", "diag"}:
		raise ValueError("mode must be one of: auto, full, diag")

	use_full = mode == "full" or (mode == "auto" and x.shape[1] <= full_max_dim)

	if use_full:
		cov_x = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1], dtype=np.float64)
		cov_y = np.cov(y, rowvar=False) + eps * np.eye(y.shape[1], dtype=np.float64)
		return float(_full_kl(mu_x, cov_x, mu_y, cov_y))

	var_x = x.var(axis=0, ddof=1) + eps
	var_y = y.var(axis=0, ddof=1) + eps
	return float(_diag_kl(mu_x, var_x, mu_y, var_y))


def compute_mmd(
	x: np.ndarray,
	y: np.ndarray,
	gamma: float = None,
	max_samples: int = 2000,
	seed: int = 0,
) -> float:
	"""MMD^2 with RBF kernel; optionally subsample for large N."""
	if x.shape[1] != y.shape[1]:
		raise ValueError("MMD requires x and y to have the same feature dimension.")

	if gamma is None:
		gamma = 1.0 / x.shape[1]

	if max_samples is not None:
		n = min(len(x), len(y), max_samples)
		if n < len(x) or n < len(y):
			rng = np.random.default_rng(seed)
			ix = rng.choice(len(x), size=n, replace=False)
			iy = rng.choice(len(y), size=n, replace=False)
			x = x[ix]
			y = y[iy]

	k_xx = rbf_kernel(x, x, gamma=gamma)
	k_yy = rbf_kernel(y, y, gamma=gamma)
	k_xy = rbf_kernel(x, y, gamma=gamma)
	return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def compute_mmd_bootstrap(
	x: np.ndarray,
	y: np.ndarray,
	n_runs: int = 5,
	sample_size: int = None,
	gamma: float = None,
	seed: int = 0,
) -> Tuple[float, float]:
	"""
	Compute MMD multiple times using bootstrap resampling.
	Returns (mean, std).
	"""
	rng = np.random.default_rng(seed)

	if sample_size is None:
		sample_size = min(len(x), len(y))

	mmd_vals = []
	for i in range(n_runs):
		ix = rng.choice(len(x), size=sample_size, replace=True)
		iy = rng.choice(len(y), size=sample_size, replace=True)
		mmd_vals.append(compute_mmd(
			x[ix],
			y[iy],
			gamma=gamma,
			max_samples=None,
			seed=seed + i,
		))

	arr = np.array(mmd_vals)
	return float(arr.mean()), float(arr.std())


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	"""Per-sample cosine distance for aligned matrices (N, D)."""
	if a.shape != b.shape:
		raise ValueError("Cosine distance requires same shape for a and b.")

	a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
	b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
	return 1.0 - np.sum(a_norm * b_norm, axis=1)


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))


# def _p(rel: str) -> str:
# 	"""Resolve a path relative to the repo root."""
# 	return os.path.join(_REPO_ROOT, rel)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument("--z_gt_path", default="../../nsdfeat/subjfeat/subj01_ave_init_latent_te.npy")
	parser.add_argument("--z_linear_path", default="../../decoded/subj01/subj01_early_scores_init_latent.npy")
	parser.add_argument("--z_mlp_path", default="../../decoded/subj01/subj01_early_scores_init_latent_mlp.npy")

	parser.add_argument("--c_gt_path", default="../../nsdfeat/subjfeat/subj01_ave_c_te.npy")
	parser.add_argument("--c_linear_path", default="../../decoded/subj01/subj01_ventral_scores_c.npy")
	parser.add_argument("--c_mlp_path", default="../../decoded/subj01/subj01_ventral_scores_c_mlp.npy")

	parser.add_argument("--quality_path", default="../../decoded/subj01/subj01_ventral_scores_c.npy")

	parser.add_argument(
		"--kl_mode",
		choices=["auto", "full", "diag"],
		default="auto",
		help="KL covariance mode. auto uses full for small D and diagonal for large D.",
	)
	parser.add_argument("--kl_full_max_dim", type=int, default=1024)
	parser.add_argument("--mmd_gamma", type=float, default=None)
	parser.add_argument("--mmd_max_samples", type=int, default=2000)
	parser.add_argument("--seed", type=int, default=0)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	args.z_gt_path = os.path.abspath(args.z_gt_path)
	args.z_linear_path = os.path.abspath(args.z_linear_path)
	args.z_mlp_path = os.path.abspath(args.z_mlp_path)
	args.c_gt_path = os.path.abspath(args.c_gt_path)
	args.c_linear_path = os.path.abspath(args.c_linear_path)
	args.c_mlp_path = os.path.abspath(args.c_mlp_path)
	args.quality_path = os.path.abspath(args.quality_path)

	z_gt, z_files = load_folder(args.z_gt_path)
	z_linear, z_linear_files = load_folder(args.z_linear_path)
	z_mlp, z_mlp_files = load_folder(args.z_mlp_path)

	c_gt, c_files = load_folder(args.c_gt_path)
	c_linear, c_linear_files = load_folder(args.c_linear_path)
	c_mlp, c_mlp_files = load_folder(args.c_mlp_path)

	assert_same_filenames(z_files, z_linear_files, "z_gt vs z_linear")
	assert_same_filenames(z_files, z_mlp_files, "z_gt vs z_mlp")
	assert_same_filenames(c_files, c_linear_files, "c_gt vs c_linear")
	assert_same_filenames(c_files, c_mlp_files, "c_gt vs c_mlp")

	if z_gt.shape[0] != c_gt.shape[0]:
		raise ValueError(
			"z and c sets have different sample counts after loading: "
			f"{z_gt.shape[0]} vs {c_gt.shape[0]}"
		)

	quality = np.load(args.quality_path)
	if quality.ndim == 1:
		quality = quality.reshape(-1)
	elif quality.ndim >= 2:
		# Reduce feature-wise quality matrices to a per-sample scalar quality.
		quality = quality.reshape(quality.shape[0], -1).mean(axis=1)

	expected_n = z_gt.shape[0]
	if quality.shape[0] != expected_n:
		raise ValueError(
			f"Quality vector length mismatch: expected {expected_n}, got {quality.shape[0]}"
		)

	print("Data loaded:")
	print(f"  z_gt      : {z_gt.shape}")
	print(f"  z_linear  : {z_linear.shape}")
	print(f"  z_mlp     : {z_mlp.shape}")
	print(f"  c_gt      : {c_gt.shape}")
	print(f"  c_linear  : {c_linear.shape}")
	print(f"  c_mlp     : {c_mlp.shape}")
	print(f"  quality   : {quality.shape}")
	print()

	print("=== KL Divergence (Gaussian) ===")
	print(
		"z_gt vs z_linear:",
		kl_gaussian(
			z_gt,
			z_linear,
			mode=args.kl_mode,
			full_max_dim=args.kl_full_max_dim,
		),
	)
	print(
		"z_gt vs z_mlp:",
		kl_gaussian(
			z_gt,
			z_mlp,
			mode=args.kl_mode,
			full_max_dim=args.kl_full_max_dim,
		),
	)
	print(
		"c_gt vs c_linear:",
		kl_gaussian(
			c_gt,
			c_linear,
			mode=args.kl_mode,
			full_max_dim=args.kl_full_max_dim,
		),
	)
	print(
		"c_gt vs c_mlp:",
		kl_gaussian(
			c_gt,
			c_mlp,
			mode=args.kl_mode,
			full_max_dim=args.kl_full_max_dim,
		),
	)

	print("\n=== MMD (RBF) ===")
	print(
		"z_gt vs z_linear:",
		compute_mmd(
			z_gt,
			z_linear,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)
	print(
		"z_gt vs z_mlp:",
		compute_mmd(
			z_gt,
			z_mlp,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)
	print(
		"c_gt vs c_linear:",
		compute_mmd(
			c_gt,
			c_linear,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)
	print(
		"c_gt vs c_mlp:",
		compute_mmd(
			c_gt,
			c_mlp,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)
	print(
		"z_linear vs z_mlp:",
		compute_mmd(
			z_linear,
			z_mlp,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)
	print(
		"c_linear vs c_mlp:",
		compute_mmd(
			c_linear,
			c_mlp,
			gamma=args.mmd_gamma,
			max_samples=args.mmd_max_samples,
			seed=args.seed,
		),
	)

	def _report_bootstrap(name: str, a: np.ndarray, b: np.ndarray) -> None:
		mean, std = compute_mmd_bootstrap(
			a,
			b,
			n_runs=5,
			sample_size=min(2000, len(a), len(b)),
			gamma=args.mmd_gamma,
			seed=args.seed,
		)
		print(f"{name}: {mean:.6f} ± {std:.6f}")

	print("\n=== MMD (Bootstrap mean ± std) ===")
	_report_bootstrap("z_gt vs z_linear", z_gt, z_linear)
	_report_bootstrap("z_gt vs z_mlp", z_gt, z_mlp)
	_report_bootstrap("c_gt vs c_linear", c_gt, c_linear)
	_report_bootstrap("c_gt vs c_mlp", c_gt, c_mlp)
	_report_bootstrap("z_linear vs z_mlp", z_linear, z_mlp)
	_report_bootstrap("c_linear vs c_mlp", c_linear, c_mlp)

	z_dist_linear = cosine_distance(z_gt, z_linear)
	z_dist_mlp = cosine_distance(z_gt, z_mlp)
	c_dist_linear = cosine_distance(c_gt, c_linear)
	c_dist_mlp = cosine_distance(c_gt, c_mlp)

	print("\n=== Correlation with Reconstruction Quality ===")
	print("z_linear:", pearsonr(z_dist_linear, quality))
	print("z_mlp:", pearsonr(z_dist_mlp, quality))
	print("c_linear:", pearsonr(c_dist_linear, quality))
	print("c_mlp:", pearsonr(c_dist_mlp, quality))


if __name__ == "__main__":
	main()
