#%%
original_data_path = '../data/nodata2_2cluster_training_set.xlsx'
simulated_data_path = '../data/nodata2_2cluster_simulated_training_patient_set_700.xlsx'

import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import linalg, stats
from matplotlib.gridspec import GridSpec


def load_align_numeric(real_path: str, synth_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load Excel files and align numeric columns present in both
    if real_path.lower().endswith('.csv'):
        df_real = pd.read_csv(real_path)
    else:
        df_real = pd.read_excel(real_path)
    if synth_path.lower().endswith('.csv'):
        df_synth = pd.read_csv(synth_path)
    else:
        df_synth = pd.read_excel(synth_path)

    num_real = df_real.select_dtypes(include=[np.number])
    num_synth = df_synth.select_dtypes(include=[np.number])

    common_cols = [c for c in num_real.columns if c in set(num_synth.columns)]
    if not common_cols:
        raise ValueError('No common numeric columns between datasets.')

    num_real = num_real[common_cols].copy()
    num_synth = num_synth[common_cols].copy()

    # Drop rows with any NaNs
    before_r, before_s = len(num_real), len(num_synth)
    num_real = num_real.dropna(axis=0, how='any')
    num_synth = num_synth.dropna(axis=0, how='any')
    after_r, after_s = len(num_real), len(num_synth)
    print(f'Real rows kept: {after_r}/{before_r}; Synthetic rows kept: {after_s}/{before_s}')

    return num_real, num_synth


def standardize_fit_transform(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    real_scaled = scaler.fit_transform(real_df.values)
    synth_scaled = scaler.transform(synth_df.values)
    real_std = pd.DataFrame(real_scaled, columns=real_df.columns, index=real_df.index)
    synth_std = pd.DataFrame(synth_scaled, columns=synth_df.columns, index=synth_df.index)
    return real_std, synth_std, scaler


def corr_and_similarity(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    # Pearson correlation matrices
    corr_a = a.corr(method='pearson')
    corr_b = b.corr(method='pearson')
    diff = corr_a - corr_b

    # Frobenius norm of difference
    frob = float(np.linalg.norm(diff.values, 'fro'))

    # Matrix correlation via vectorized upper triangles (exclude diagonal)
    p = corr_a.shape[0]
    iu = np.triu_indices(p, k=1)
    va = corr_a.values[iu]
    vb = corr_b.values[iu]
    if va.size == 0:
        mat_corr = float('nan')
    else:
        r, _ = stats.pearsonr(va, vb)
        mat_corr = float(r)
    return corr_a, corr_b, diff, frob, mat_corr


def save_heatmap(matrix: pd.DataFrame, title: str, out_path: str, vmin: Optional[float] = -1.0, vmax: Optional[float] = 1.0) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix.values, vmin=vmin, vmax=vmax, cmap='coolwarm', aspect='auto')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def hotellings_t2_two_sample(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, int, int, float]:
    # Two-sample Hotelling's T^2 test with pooled covariance
    n1, p = X.shape
    n2, _ = Y.shape
    mean1 = X.mean(axis=0)
    mean2 = Y.mean(axis=0)
    d = mean1 - mean2
    S1 = np.cov(X, rowvar=False, bias=False)
    S2 = np.cov(Y, rowvar=False, bias=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    # Invert with pinv for numerical stability
    try:
        Sp_inv = linalg.inv(Sp)
    except linalg.LinAlgError:
        # Fallback to NumPy's pinv to avoid SciPy's rcond signature mismatch
        Sp_inv = np.linalg.pinv(Sp, rcond=1e-10)
    T2 = float((n1 * n2) / (n1 + n2) * (d.T @ Sp_inv @ d))

    # F approximation
    df1 = p
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        F = float('nan')
        pval = float('nan')
    else:
        F = float((df2 / (df1 * (n1 + n2 - 2))) * T2)
        pval = float(1.0 - stats.f.cdf(F, df1, df2))
    return T2, F, df1, df2, pval


def mardia_tests(X: np.ndarray) -> Tuple[float, float, int, float, float]:
    # Returns: b1p (skewness), skew_chi2, skew_df, b2p (kurtosis), kurt_z p-value
    n, p = X.shape
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, bias=False)
    # Regularize if singular
    try:
        S_inv = linalg.inv(S)
    except linalg.LinAlgError:
        S_inv = np.linalg.pinv(S, rcond=1e-10)

    Xc = X - mu
    # Whiten to Z ~ standardized with covariance ~ I
    # Compute S^{-1/2} via eigen-decomposition for stability
    evals, evecs = linalg.eigh(S)
    evals[evals < 1e-12] = 1e-12
    S_inv_sqrt = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T)
    Z = Xc @ S_inv_sqrt

    # Mardia skewness: b1p = (1/n^2) sum_{i,j} (z_i^T z_j)^3
    # Compute in blocks to avoid O(n^2) memory when n large
    def sum_cubes_gram(Z: np.ndarray, block: int = 2000) -> float:
        n_local = Z.shape[0]
        total = 0.0
        for i in range(0, n_local, block):
            Zi = Z[i:i + block]
            # compute Zi @ Z^T in blocks over columns to control memory
            for j in range(0, n_local, block):
                Zj = Z[j:j + block]
                G = Zi @ Zj.T  # (bi x bj)
                total += np.sum(G * G * G)
        return total

    sum_cubes = sum_cubes_gram(Z)
    b1p = float(sum_cubes / (n * n))
    skew_df = int(p * (p + 1) * (p + 2) // 6)
    skew_chi2 = float(n * b1p / 6.0)
    skew_p = float(1.0 - stats.chi2.cdf(skew_chi2, df=skew_df))

    # Mardia kurtosis: b2p = (1/n) sum_i (z_i^T z_i)^2
    di2 = np.sum(Z * Z, axis=1)
    b2p = float(np.mean(di2 ** 2))
    mean_b2 = p * (p + 2)
    var_b2 = 8.0 * p * (p + 2) / n
    kurt_z = float((b2p - mean_b2) / np.sqrt(var_b2))
    kurt_p = float(2.0 * (1.0 - stats.norm.cdf(abs(kurt_z))))

    return b1p, skew_p, skew_df, b2p, kurt_p


def try_pingouin_hotelling_mardia(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    try:
        import pingouin as pg  # type: ignore
    except Exception:
        return None
    results = {}
    try:
        # Hotelling two-sample
        # pingouin expects a long format for multivariate? It has multivariate_ttest for one-sample.
        # Use fallback if unavailable; keep only Mardia from pg
        pass
    except Exception:
        pass
    return None


def plot_tsne_with_marginals(emb: np.ndarray, labels: np.ndarray, pval: float, out_dir: str,
                             title: str = 't-SNE: Real vs Synthetic with Marginal Densities') -> None:
    # Prepare masks
    mask_r = (labels == 'real')
    mask_s = (labels == 'synthetic')

    x = emb[:, 0]
    y = emb[:, 1]
    xr, yr = x[mask_r], y[mask_r]
    xs, ys = x[mask_s], y[mask_s]

    def padded_limits(arr: np.ndarray, pad_frac: float = 0.05):
        a_min = float(np.min(arr))
        a_max = float(np.max(arr))
        if not np.isfinite(a_min) or not np.isfinite(a_max):
            return (-1.0, 1.0)
        if a_min == a_max:
            pad = 1.0
            return a_min - pad, a_max + pad
        pad = (a_max - a_min) * pad_frac
        return a_min - pad, a_max + pad

    xlim = padded_limits(x)
    ylim = padded_limits(y)

    # KDE helpers
    def safe_kde(data: np.ndarray):
        if data.size < 2 or np.allclose(data, data[0]):
            return lambda z: np.zeros_like(z, dtype=float)
        try:
            kde = stats.gaussian_kde(data)
        except Exception:
            return lambda z: np.zeros_like(z, dtype=float)
        return kde.evaluate

    nx = 200
    ny = 200
    x_grid = np.linspace(xlim[0], xlim[1], nx)
    y_grid = np.linspace(ylim[0], ylim[1], ny)

    dx_r = safe_kde(xr)(x_grid)
    dx_s = safe_kde(xs)(x_grid)
    dy_r = safe_kde(yr)(y_grid)
    dy_s = safe_kde(ys)(y_grid)

    xmax = max(dx_r.max() if dx_r.size else 0.0, dx_s.max() if dx_s.size else 0.0) or 1.0
    ymax = max(dy_r.max() if dy_r.size else 0.0, dy_s.max() if dy_s.size else 0.0) or 1.0

    fig = plt.figure(figsize=(8.0, 6.5))
    gs = GridSpec(nrows=2, ncols=2, figure=fig,
                  width_ratios=[4.0, 1.2], height_ratios=[1.2, 4.0],
                  wspace=0.05, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    ax_main.scatter(xr, yr, s=12, c='tab:blue', label='Real', alpha=0.7)
    ax_main.scatter(xs, ys, s=12, c='tab:orange', label='Synthetic', alpha=0.7, marker='x')
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_xlabel('t-SNE 1')
    ax_main.set_ylabel('t-SNE 2')
    ax_main.grid(False)
    ax_main.legend(frameon=False, loc='best')

    # Annotate p-value at the bottom-right of the figure
    p_text = f"Hotelling's T² p = {pval:.3e}"
    fig.text(0.98, 0.02, p_text,
             ha='right', va='top', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Top marginal KDE
    ax_top.plot(x_grid, dx_r, color='tab:blue', lw=1.6)
    ax_top.plot(x_grid, dx_s, color='tab:orange', lw=1.6)
    ax_top.set_xlim(xlim)
    ax_top.set_ylim(0, xmax * 1.05)
    ax_top.set_xticks([])
    ax_top.set_ylabel('Density')
    ax_top.grid(False)

    # Right marginal KDE (horizontal)
    ax_right.plot(dy_r, y_grid, color='tab:blue', lw=1.6)
    ax_right.plot(dy_s, y_grid, color='tab:orange', lw=1.6)
    ax_right.set_ylim(ylim)
    ax_right.set_xlim(0, ymax * 1.05)
    ax_right.set_yticks([])
    ax_right.set_xlabel('Density')
    ax_right.grid(False)

    # Style spines
    for ax in (ax_top, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in (ax_top, ax_main, ax_right):
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.suptitle(title, y=0.98, fontsize=12)
    plt.tight_layout()

    png_path = os.path.join(out_dir, 'tsne_with_marginals.png')
    pdf_path = os.path.join(out_dir, 'tsne_with_marginals.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'Results')
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load and align numeric columns
    real_df, synth_df = load_align_numeric(original_data_path, simulated_data_path)

    # 2) Standardize (fit on real)
    real_std, synth_std, _ = standardize_fit_transform(real_df, synth_df)
    # Drop zero-variance columns in either dataset to avoid NaNs in correlations
    tol = 1e-12
    std_real = real_std.std(axis=0, ddof=1)
    std_synth = synth_std.std(axis=0, ddof=1)
    keep_cols = (std_real > tol) & (std_synth > tol)
    if not np.all(keep_cols):
        dropped = list(real_std.columns[~keep_cols])
        print(f'Dropping {len(dropped)} constant/near-constant columns: {dropped[:8]}{"..." if len(dropped) > 8 else ""}')
        real_std = real_std.loc[:, keep_cols]
        synth_std = synth_std.loc[:, keep_cols]
    if real_std.shape[1] < 2:
        raise ValueError('Not enough non-constant numeric features after filtering to compute correlations (need at least 2).')

    real_std_path = os.path.join(out_dir, 'standardized_real.csv')
    synth_std_path = os.path.join(out_dir, 'standardized_synth.csv')
    real_std.to_csv(real_std_path, index=False)
    synth_std.to_csv(synth_std_path, index=False)

    # 3) Correlation analysis
    corr_real, corr_synth, corr_diff, frob, mat_corr = corr_and_similarity(real_std, synth_std)
    corr_real.to_csv(os.path.join(out_dir, 'corr_real.csv'))
    corr_synth.to_csv(os.path.join(out_dir, 'corr_synth.csv'))
    corr_diff.to_csv(os.path.join(out_dir, 'corr_diff.csv'))

    save_heatmap(corr_real, 'Correlation (Real)', os.path.join(out_dir, 'corr_real.png'))
    save_heatmap(corr_synth, 'Correlation (Synthetic)', os.path.join(out_dir, 'corr_synth.png'))
    # For difference, set limits to [-2,2] since correlations in [-1,1]
    save_heatmap(corr_diff, 'Correlation Difference (Real - Synthetic)', os.path.join(out_dir, 'corr_diff.png'), vmin=-2.0, vmax=2.0)

    with open(os.path.join(out_dir, 'correlation_metrics.txt'), 'w') as f:
        f.write('Correlation structure comparison\n')
        f.write(f'Frobenius norm of difference: {frob:.6f}\n')
        f.write(f'Upper-triangle matrix correlation: {mat_corr:.6f}\n')

    # 4) Multivariate tests
    X = real_std.values
    Y = synth_std.values
    T2, F, df1, df2, pval = hotellings_t2_two_sample(X, Y)

    # Mardia for real and synth
    b1p_r, skew_p_r, skew_df_r, b2p_r, kurt_p_r = mardia_tests(X)
    b1p_s, skew_p_s, skew_df_s, b2p_s, kurt_p_s = mardia_tests(Y)

    with open(os.path.join(out_dir, 'multivar_tests.txt'), 'w') as f:
        f.write('# Multivariate Tests\n')
        f.write('## Hotelling T^2 (two-sample)\n')
        f.write(f'T2: {T2:.6f}\n')
        f.write(f'F approx: {F:.6f}\n')
        f.write(f'df1: {df1}, df2: {df2}\n')
        f.write(f'p-value: {pval:.6e}\n')
        f.write('\n## Mardia multivariate normality (Real)\n')
        f.write(f'Skewness b1p: {b1p_r:.6f}, p-value: {skew_p_r:.6e}, df: {skew_df_r}\n')
        f.write(f'Kurtosis b2p: {b2p_r:.6f}, p-value (z-test): {kurt_p_r:.6e}\n')
        f.write('\n## Mardia multivariate normality (Synthetic)\n')
        f.write(f'Skewness b1p: {b1p_s:.6f}, p-value: {skew_p_s:.6e}, df: {skew_df_s}\n')
        f.write(f'Kurtosis b2p: {b2p_s:.6f}, p-value (z-test): {kurt_p_s:.6e}\n')

    # 5) t-SNE visualization
    n_real = X.shape[0]
    n_synth = Y.shape[0]
    XY = np.vstack([X, Y])
    labels = np.array(['real'] * n_real + ['synthetic'] * n_synth)
    n_total = XY.shape[0]
    perplexity = int(max(5, min(30, (n_total - 1) // 3)))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
    emb = tsne.fit_transform(XY)

    plt.figure(figsize=(7, 5))
    mask_r = labels == 'real'
    mask_s = labels == 'synthetic'
    plt.scatter(emb[mask_r, 0], emb[mask_r, 1], s=12, c='tab:blue', label='Real', alpha=0.7)
    plt.scatter(emb[mask_s, 0], emb[mask_s, 1], s=12, c='tab:orange', label='Synthetic', alpha=0.7, marker='x')
    plt.title(f't-SNE: Real vs Synthetic (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tsne_real_vs_synth.png'), dpi=200)
    plt.close()

    # Also produce scatter with marginal KDEs and p-value annotation (PNG + PDF)
    plot_tsne_with_marginals(emb, labels, pval, out_dir)

    # Print summary
    print('--- Summary ---')
    print(f'Correlation matrix correlation (upper triangle): {mat_corr:.4f}')
    print(f'Correlation Frobenius norm difference: {frob:.4f}')
    print(f"Hotelling's T^2 p-value: {pval:.3e}")
    print(f'Mardia skew p (real, synth): {skew_p_r:.3e}, {skew_p_s:.3e}')
    print(f'Mardia kurt p (real, synth): {kurt_p_r:.3e}, {kurt_p_s:.3e}')


if __name__ == '__main__':
    main()

# %%
