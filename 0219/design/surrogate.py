"""
Surrogate Model: Gaussian Process Regression

Trains GP surrogate models on DOE data for fast optimization.

Target outputs (3 objectives):
  1) P_total    : Hover power (minimize)
  2) FM         : Figure of Merit (maximize)
  3) T_margin_pct : Thrust margin at rho_min (maximize, robustness)

Usage:
  python 0219/design/surrogate.py               # Train from doe_results.json
  python 0219/design/surrogate.py --doe-file results/doe/doe_results.json
"""

import argparse
import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DESIGN_LOWER, DESIGN_UPPER, N_DESIGN_VARS, DESIGN_VAR_NAMES,
    RESULTS_DIR,
)
from utils import ensure_results_dir, save_json


# Surrogate targets
TARGETS = ['P_total', 'FM', 'T_margin_pct_rhomin']
TARGET_LABELS = ['Hover Power (W)', 'Figure of Merit', 'T margin @ rho_min (%)']


def load_doe_data(doe_file: str, feasible_only: bool = True):
    """
    Load DOE results and extract X, Y matrices.

    Returns:
      X : (N, 12) design vectors (normalized [0,1])
      Y : (N, 3)  target KPIs [P_total, FM, T_margin_pct_rhomin]
      raw_results : list of dicts
    """
    with open(doe_file, 'r') as f:
        doe_output = json.load(f)

    results = doe_output['results']

    if feasible_only:
        results = [r for r in results if r.get('feasible', False)]

    if len(results) == 0:
        raise ValueError("No feasible samples found in DOE data!")

    X_raw = np.array([r['design_vector'] for r in results])
    Y = np.zeros((len(results), len(TARGETS)))

    for i, r in enumerate(results):
        for j, target in enumerate(TARGETS):
            val = r.get(target, np.nan)
            Y[i, j] = val if val is not None else np.nan

    # Normalize X to [0, 1]
    X = (X_raw - DESIGN_LOWER) / (DESIGN_UPPER - DESIGN_LOWER)
    X = np.clip(X, 0.0, 1.0)

    # Remove rows with NaN in Y
    valid = ~np.any(np.isnan(Y), axis=1)
    X = X[valid]
    Y = Y[valid]
    results = [results[i] for i in range(len(results)) if valid[i]]

    print(f"  Loaded {len(results)} valid samples from DOE")
    return X, Y, results


def build_gp_models(X: np.ndarray, Y: np.ndarray, verbose: bool = True):
    """
    Train one GP model per target output.

    Returns list of trained GP regressors.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    models = []

    for j, (target, label) in enumerate(zip(TARGETS, TARGET_LABELS)):
        y_j = Y[:, j]

        # Kernel: Constant * Matern(nu=2.5) + White noise
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(X.shape[1]),
                     length_scale_bounds=(1e-2, 10.0), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        gp.fit(X, y_j)

        # Leave-One-Out cross-validation (approximation via prediction)
        y_pred = gp.predict(X)
        loo_rmse = float(np.sqrt(np.mean((y_pred - y_j) ** 2)))

        models.append(gp)

        if verbose:
            print(f"  GP [{target}]:")
            print(f"    Train RMSE: {loo_rmse:.4f}  |  "
                  f"log-likelihood: {gp.log_marginal_likelihood_value_:.3f}")

    return models


def run_surrogate(doe_file: str = None, feasible_only: bool = True):
    print("=" * 70)
    print("Surrogate Model: Gaussian Process Regression")
    print("=" * 70)

    results_dir = ensure_results_dir("surrogate")

    if doe_file is None:
        doe_file = str(RESULTS_DIR / "doe" / "doe_results.json")

    if not Path(doe_file).exists():
        print(f"ERROR: DOE file not found: {doe_file}")
        print("  Run doe.py first: python 0219/design/doe.py")
        return None

    print(f"  DOE file: {doe_file}")
    print(f"  Targets: {TARGETS}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print("[Loading DOE data...]")
    X, Y, raw_results = load_doe_data(doe_file, feasible_only=feasible_only)
    print(f"  X shape: {X.shape}  Y shape: {Y.shape}")
    print()

    # ── Data statistics ───────────────────────────────────────────────────
    print("[Target Statistics]")
    for j, (target, label) in enumerate(zip(TARGETS, TARGET_LABELS)):
        yj = Y[:, j]
        print(f"  {target:28s}: "
              f"min={yj.min():.4f}  max={yj.max():.4f}  "
              f"mean={yj.mean():.4f}  std={yj.std():.4f}")
    print()

    # ── Train GP models ───────────────────────────────────────────────────
    print("[Training GP models...]")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return None

    models = build_gp_models(X, Y, verbose=True)
    print()

    # ── Cross-validation (hold-out 20%) ───────────────────────────────────
    print("[Cross-validation (20% hold-out)]")
    n = len(X)
    n_test = max(1, n // 5)
    idx = np.random.default_rng(42).permutation(n)
    X_train, X_test = X[idx[n_test:]], X[idx[:n_test]]
    Y_train, Y_test = Y[idx[n_test:]], Y[idx[:n_test]]

    cv_errors = {}
    for j, target in enumerate(TARGETS):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        kernel = (
            ConstantKernel(1.0) * Matern(nu=2.5)
            + WhiteKernel(1e-4)
        )
        gp_cv = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=True
        )
        gp_cv.fit(X_train, Y_train[:, j])
        y_pred = gp_cv.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_pred - Y_test[:, j]) ** 2)))
        r2 = float(1.0 - np.sum((y_pred - Y_test[:, j]) ** 2) /
                   np.sum((Y_test[:, j] - Y_test[:, j].mean()) ** 2))
        cv_errors[target] = {'rmse': rmse, 'r2': r2}
        print(f"  {target:28s}: RMSE={rmse:.4f}  R²={r2:.4f}")

    print()

    # ── Save models ────────────────────────────────────────────────────────
    model_bundle = {
        'models': models,
        'targets': TARGETS,
        'target_labels': TARGET_LABELS,
        'X_train': X,
        'Y_train': Y,
        'design_var_names': DESIGN_VAR_NAMES,
        'design_lower': DESIGN_LOWER,
        'design_upper': DESIGN_UPPER,
        'n_samples': len(X),
        'cv_errors': cv_errors,
    }

    model_path = str(results_dir / "gp_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"  Model saved: {model_path}")

    # Save metadata JSON
    meta = {
        'n_samples': len(X),
        'targets': TARGETS,
        'cv_errors': cv_errors,
        'feasible_only': feasible_only,
        'doe_file': doe_file,
    }
    save_json(str(results_dir / "surrogate_meta.json"), meta)

    print(f"\nResults saved to: {results_dir}")
    return model_bundle


def load_surrogate(model_path: str = None):
    """Load trained surrogate model bundle."""
    if model_path is None:
        model_path = str(RESULTS_DIR / "surrogate" / "gp_model.pkl")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict(model_bundle: dict, design_vector: np.ndarray) -> dict:
    """
    Predict KPIs for a new design vector.

    Parameters
    ----------
    model_bundle : dict  from load_surrogate()
    design_vector : (12,) array in physical units

    Returns
    -------
    dict with predicted KPIs and uncertainties
    """
    lower = model_bundle['design_lower']
    upper = model_bundle['design_upper']
    x_norm = (design_vector - lower) / (upper - lower)
    x_norm = np.clip(x_norm, 0.0, 1.0).reshape(1, -1)

    result = {}
    for j, (model, target) in enumerate(
            zip(model_bundle['models'], model_bundle['targets'])):
        y_pred, y_std = model.predict(x_norm, return_std=True)
        result[target] = float(y_pred[0])
        result[target + '_std'] = float(y_std[0])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate: GP Regression")
    parser.add_argument("--doe-file", type=str, default=None,
                        help="Path to doe_results.json")
    parser.add_argument("--all", action="store_true",
                        help="Use all samples (not just feasible)")
    args = parser.parse_args()

    run_surrogate(
        doe_file=args.doe_file,
        feasible_only=not args.all,
    )
