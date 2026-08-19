#!/usr/bin/env python3
"""
train_wgs_depth_model.py

Trains a tree-based model on the training_data.parquet produced by
prepare_wgs_depth_features.py, to predict WGS tumor depth from WES raw
depth, GC%, distance-to-target, and (optionally) genomic location.

ROBUST OBJECTIVE (--objective)
----------------------------------
Because the same genomic bin can be recurrently amplified in several of the
pooled samples, a plain squared-error (l2) fit gets pulled toward those
elevated values -- the model partly "learns to expect" the amplification at
that location, which is exactly what you don't want if the eventual use is
computing a per-sample residual (observed - predicted) as an amplification
signal. Options here:

  l2       standard squared-error regression (LightGBM default / sklearn RF).
  huber    LightGBM's Huber loss -- quadratic near zero, linear (so much less
           sensitive to outliers) beyond --huber-alpha. Good middle ground:
           still uses all the data, just down-weights the influence of large
           residuals rather than chasing them.
  quantile median regression (LightGBM objective='quantile', alpha=0.5).
           Directly fits the conditional median rather than the mean, which
           is naturally robust to a minority of high-depth (amplified)
           samples at a given bin pulling the fit upward.

For --model rf, huber/quantile aren't natively available in scikit-learn's
RandomForestRegressor; --objective huber or quantile is approximated there
via criterion='absolute_error' (L1 / median-seeking splits), which is the
closest available robust option, and a note is printed to that effect.

CROSS-VALIDATION
------------------
--cv-group sample (default): leave-samples-out. Tests generalization to a
    new tumor at bins already seen during training.
--cv-group chrom: leave-chromosomes-out. Tests generalization to entirely
    unseen genomic regions -- the honest check on whether location features
    (especially bin-id) are learning something transferable.

SCORING BELOW --min-wes-depth (--score-full-wes-range)
----------------------------------------------------------
If prepare_wgs_depth_features.py was run with --save-full-wes-range, it
wrote full_wes_range_data.parquet alongside the usual training_data.parquet
-- every bin --min-wes-depth would normally have dropped, still carrying
mask_col/--min-wgs-depth/dropna filtering, tagged 'below_min_wes_depth'.
Pass --score-full-wes-range here to have each fold's model ALSO predict on
that fold's held-out samples' (or chroms') below-threshold bins -- never
used for training, only scored, using the same OOF discipline as the main
loop (a bin from a held-out group is always scored by a model that never
saw that group). Written to cv_predictions_recovered_low_wes_depth.csv.

CAUTION: the model never saw raw_wes_depth this low during training. Tree
models don't extrapolate smoothly outside the training range -- these rows
get routed into whichever leaf covered the lowest depths the model DID see,
not something calibrated for depth this low. Treat these predictions as
lower-confidence coverage recovery, not equivalent-quality data; the output
file is kept separate from cv_predictions.csv for exactly this reason,
rather than silently merged in.

--zero-predict-wes-depth-threshold goes a step further for the very bottom
of that range: below the given raw_wes_depth value, even leaf-based
extrapolation is skipped entirely and oof_prediction is hardcoded to 0.0.
These rows are still written to the output (never dropped) -- only the
prediction value changes, from "whatever leaf a near-zero-coverage bin
happens to land in" to a defined, non-extrapolated 0.

OUTPUT (written to --outdir)
-------------------------------
  model.joblib                  final model trained on all provided data
  cv_predictions.csv             out-of-fold predictions for every row
  cv_predictions_recovered_low_wes_depth.csv
                                  (only with --score-full-wes-range) OOF
                                  predictions for bins below --min-wes-depth,
                                  scored but never trained on -- see caution
                                  above before treating these as equal-
                                  confidence to cv_predictions.csv
  cv_metrics_per_fold.csv        R2 / MAE / Spearman rho per fold
  cv_metrics_summary.txt         aggregated CV metrics + run parameters
  feature_importance.csv/.png    final-model feature importances
  predicted_vs_actual.png        held-out predicted vs actual WGS depth
  per_sample_metrics.csv         held-out R2/MAE broken out by sample
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_prepared_data(data_dir):
    data_dir = Path(data_dir)
    parquet_path = data_dir / 'training_data.parquet'
    metadata_path = data_dir / 'metadata.json'
    if not parquet_path.exists() or not metadata_path.exists():
        sys.exit(f"Error: expected {parquet_path} and {metadata_path} -- "
                  f"run prepare_wgs_depth_features.py first.")

    df = pd.read_parquet(parquet_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    for col in metadata['categorical_cols']:
        if col in df.columns and not str(df[col].dtype) == 'category':
            df[col] = df[col].astype('category')

    return df, metadata


def load_full_wes_range_data(data_dir, metadata):
    """Loads full_wes_range_data.parquet (from prepare_wgs_depth_features.py
    --save-full-wes-range), applying the same categorical dtype fix-up as
    load_prepared_data(). Returns None if the flag wasn't used upstream --
    caller decides whether that's an error or just 'nothing to score'."""
    if not metadata.get('has_full_wes_range_data'):
        return None
    data_dir = Path(data_dir)
    fname = metadata.get('full_wes_range_parquet') or 'full_wes_range_data.parquet'
    parquet_path = data_dir / fname
    if not parquet_path.exists():
        print(f"    [-] Warning: metadata.json says full_wes_range_data.parquet should exist "
              f"but {parquet_path} isn't there -- skipping --score-full-wes-range.")
        return None

    df_full = pd.read_parquet(parquet_path)
    for col in metadata['categorical_cols']:
        if col in df_full.columns and not str(df_full[col].dtype) == 'category':
            df_full[col] = df_full[col].astype('category')
    return df_full


def one_hot_for_rf(X, categorical_cols):
    """Random Forest has no native categorical support -- one-hot encode."""
    return pd.get_dummies(X, columns=categorical_cols, dummy_na=False)


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def make_model(model_name, objective, huber_alpha, quantile_alpha, n_jobs):
    if model_name == 'lgbm':
        from lightgbm import LGBMRegressor
        kwargs = dict(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=n_jobs,
        )
        if objective == 'l2':
            kwargs['objective'] = 'regression'
        elif objective == 'huber':
            kwargs['objective'] = 'huber'
            kwargs['alpha'] = huber_alpha
        elif objective == 'quantile':
            kwargs['objective'] = 'quantile'
            kwargs['alpha'] = quantile_alpha
        else:
            raise ValueError(f"Unknown objective: {objective}")
        return LGBMRegressor(**kwargs)

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        if objective == 'l2':
            criterion = 'squared_error'
        else:
            criterion = 'absolute_error'
            print(f"    [-] Note: RandomForestRegressor has no huber/quantile objective; "
                  f"approximating --objective {objective} with criterion='absolute_error' "
                  f"(L1 / median-seeking splits).")
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            criterion=criterion,
            random_state=42,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def fit_predict_multi(model_name, X_train, y_train, X_test_list, categorical_cols,
                       objective, huber_alpha, quantile_alpha, n_jobs):
    """Fits ONE model instance and returns a list of predictions, one per
    entry in X_test_list (None entries pass through as None). Used so the
    same fold's model can score both its normal held-out test rows AND that
    fold's held-out groups' recovered low-WES-depth rows, without paying
    the fit cost twice."""
    model = make_model(model_name, objective, huber_alpha, quantile_alpha, n_jobs)

    if model_name == 'lgbm':
        model.fit(X_train, y_train, categorical_feature=categorical_cols or 'auto')
        return [None if Xt is None else model.predict(Xt) for Xt in X_test_list]

    # RF has no native categorical support -- one-hot everything against a
    # SHARED column set (union across train + every test frame) so all
    # frames align to the same columns the single fitted model expects.
    X_train_oh = one_hot_for_rf(X_train, categorical_cols)
    test_oh_list = [None if Xt is None else one_hot_for_rf(Xt, categorical_cols) for Xt in X_test_list]
    all_cols = X_train_oh.columns
    for t in test_oh_list:
        if t is not None:
            all_cols = all_cols.union(t.columns)
    X_train_oh = X_train_oh.reindex(columns=all_cols, fill_value=0)
    model.fit(X_train_oh, y_train)
    preds = []
    for t in test_oh_list:
        if t is None:
            preds.append(None)
        else:
            preds.append(model.predict(t.reindex(columns=all_cols, fill_value=0)))
    return preds


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def get_log_axis_limits(*arrays, pad_factor=1.2, min_floor=1e-1):
    positive_vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            positive_vals.append(a)
    if not positive_vals:
        return min_floor, min_floor * 10
    combined = np.concatenate(positive_vals)
    lo = max(np.nanmin(combined) / pad_factor, min_floor)
    hi = np.nanmax(combined) * pad_factor
    return lo, hi


def plot_predicted_vs_actual(y_true, y_pred, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=3, alpha=0.15, color='#4C72B0', linewidths=0,
               rasterized=True)

    x_min, x_max = get_log_axis_limits(y_true, y_pred)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.plot([x_min, x_max], [x_min, x_max], color='gray', linestyle='--',
             linewidth=1.2, label='1:1 (perfect prediction)')

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    r2 = r2_score(y_true[valid], y_pred[valid]) if valid.sum() > 1 else np.nan
    rho = spearmanr(y_true[valid], y_pred[valid]).correlation if valid.sum() > 1 else np.nan

    ax.set_xlabel('Actual WGS tumor depth', fontsize=9)
    ax.set_ylabel('Predicted WGS tumor depth', fontsize=9)
    ax.set_title(f'{title}\nR2={r2:.3f}   Spearman rho={rho:.3f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path):
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        return
    order = np.argsort(importances)[::-1]
    top = order[:30]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(top))))
    ax.barh(np.array(feature_names)[top][::-1], np.array(importances)[top][::-1],
            color='#55A868')
    ax.set_xlabel('Importance', fontsize=9)
    ax.set_title('Feature importance (final model, all data)', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True,
                   help='Directory containing training_data.parquet + metadata.json, '
                        'as produced by prepare_wgs_depth_features.py')
    p.add_argument('--model', choices=['lgbm', 'rf'], default='lgbm',
                   help='Tree-based model to train. Default: lgbm')
    p.add_argument('--objective', choices=['l2', 'huber', 'quantile'], default='l2',
                   help="Loss function. 'huber' and 'quantile' are more robust to a minority "
                        "of recurrently-amplified samples pulling the fit upward at a given "
                        "bin. See module docstring. Default: l2")
    p.add_argument('--huber-alpha', type=float, default=1.0,
                   help='Huber loss delta threshold (LightGBM alpha param for objective=huber). '
                        'Below this residual magnitude, loss is quadratic; above it, linear. '
                        'Default: 1.0')
    p.add_argument('--quantile-alpha', type=float, default=0.5,
                   help='Quantile to fit when --objective quantile. 0.5 = median regression. '
                        'Default: 0.5')
    p.add_argument('--cv-group', choices=['sample', 'chrom'], default='sample',
                   help="Grouping for cross-validation. 'sample' = leave-samples-out. "
                        "'chrom' = leave-chromosomes-out. Default: sample")
    p.add_argument('--n-splits', type=int, default=5,
                   help='Number of CV folds, capped at the number of distinct groups. Default: 5')
    p.add_argument('--log-target', action='store_true', default=True,
                   help='Fit log1p(wgs_tumor_depth) instead of raw depth (default: on). '
                        'Pass --no-log-target to disable.')
    p.add_argument('--no-log-target', dest='log_target', action='store_false')
    p.add_argument('--n-jobs', type=int, default=-1, help='Threads for model fitting. Default: -1')
    p.add_argument('--score-full-wes-range', action='store_true',
                   help="If prepare_wgs_depth_features.py was run with --save-full-wes-range, "
                        "also score each fold's held-out samples'/chroms' below-min-wes-depth "
                        "bins with that fold's model (never trained on), written to "
                        "cv_predictions_recovered_low_wes_depth.csv. See module docstring for "
                        "the extrapolation caveat. No-op with a warning if the upstream parquet "
                        "isn't present.")
    p.add_argument('--zero-predict-wes-depth-threshold', type=float, default=None,
                   help="Within the --score-full-wes-range recovered output, bins with "
                        "raw_wes_depth <= this value get oof_prediction hardcoded to 0.0 "
                        "instead of being scored by the model. A model that never trained on "
                        "genuinely zero/near-zero WES coverage has no real basis to extrapolate "
                        "there, so this substitutes a defined answer instead of a leaf-based "
                        "guess. These rows are still WRITTEN to "
                        "cv_predictions_recovered_low_wes_depth.csv, never dropped -- this only "
                        "changes what value they get. e.g. --zero-predict-wes-depth-threshold 0 "
                        "hardcodes exactly-zero-coverage bins to 0 while still letting the model "
                        "score everything else in the recovered range. Default: None (score "
                        "everything with the model).")
    p.add_argument('--outdir', required=True, help='Output directory')
    args = p.parse_args()

    df, metadata = load_prepared_data(args.data_dir)
    print(f"[+] Loaded {len(df):,} rows across {metadata['n_samples']} samples "
          f"(location_encoding={metadata['location_encoding']})")

    if metadata['location_encoding'] == 'bin-id' and args.model != 'lgbm':
        sys.exit("Error: this data was prepared with --location-encoding bin-id, which "
                  "requires --model lgbm (a Random Forest would need it one-hot encoded, "
                  "impractical at this cardinality). Re-run prepare_wgs_depth_features.py "
                  "with --location-encoding coords for an rf-compatible dataset.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_cols = metadata['feature_cols']
    categorical_cols = metadata['categorical_cols']
    X = df[feature_cols].copy()
    y_raw = df['y_wgs_tumor_depth'].values.astype(float)
    y = np.log1p(y_raw) if args.log_target else y_raw
    groups = df['meta_sample'].values if args.cv_group == 'sample' else df['meta_chrom'].values

    df_full = None
    if args.score_full_wes_range:
        df_full = load_full_wes_range_data(args.data_dir, metadata)
        if df_full is None:
            print("    [-] --score-full-wes-range was set but no full_wes_range_data.parquet "
                  "is available -- continuing without it.")
        else:
            missing_full_cols = set(feature_cols) - set(df_full.columns)
            if missing_full_cols:
                sys.exit(f"Error: full_wes_range_data.parquet is missing feature columns "
                          f"{missing_full_cols} that training_data.parquet has -- was it built "
                          f"by a matching run of prepare_wgs_depth_features.py?")
            X_full = df_full[feature_cols].copy()
            y_full_raw = df_full['y_wgs_tumor_depth'].values.astype(float)
            groups_full = (df_full['meta_sample'].values if args.cv_group == 'sample'
                            else df_full['meta_chrom'].values)
            oof_pred_full = np.full(len(df_full), np.nan)
            below_mask_full = df_full['below_min_wes_depth'].values.astype(bool)
            n_recovered = int(below_mask_full.sum())
            print(f"[+] Loaded full_wes_range_data.parquet: {len(df_full):,} rows, "
                  f"{n_recovered:,} below --min-wes-depth (will be scored per-fold, never trained on)")

    n_groups = len(np.unique(groups))
    n_splits = min(args.n_splits, n_groups)
    if n_splits < 2:
        sys.exit(f"Error: need at least 2 groups for --cv-group {args.cv_group}, found {n_groups}.")
    print(f"[+] Cross-validating with GroupKFold(n_splits={n_splits}), grouped by '{args.cv_group}' "
          f"({n_groups} distinct groups), objective='{args.objective}'...")

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(df), np.nan)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        held_out_groups = np.unique(groups[test_idx])

        full_held_idx = None
        X_full_held = None
        if df_full is not None:
            full_held_mask = np.isin(groups_full, held_out_groups) & below_mask_full
            if args.zero_predict_wes_depth_threshold is not None:
                zero_mask = full_held_mask & (df_full['raw_wes_depth'].values <= args.zero_predict_wes_depth_threshold)
                score_mask = full_held_mask & ~zero_mask
            else:
                zero_mask = np.zeros(len(df_full), dtype=bool)
                score_mask = full_held_mask

            zero_idx = np.where(zero_mask)[0]
            if len(zero_idx):
                oof_pred_full[zero_idx] = 0.0

            full_held_idx = np.where(score_mask)[0]
            if len(full_held_idx):
                X_full_held = X_full.iloc[full_held_idx]

        pred_log, pred_full_log = fit_predict_multi(
            args.model, X_train, y_train, [X_test, X_full_held], categorical_cols,
            args.objective, args.huber_alpha, args.quantile_alpha, args.n_jobs)
        pred = np.expm1(pred_log) if args.log_target else pred_log
        oof_pred[test_idx] = pred

        if pred_full_log is not None:
            pred_full = np.expm1(pred_full_log) if args.log_target else pred_full_log
            oof_pred_full[full_held_idx] = pred_full

        y_test_raw = y_raw[test_idx]
        r2 = r2_score(y_test_raw, pred)
        mae = mean_absolute_error(y_test_raw, pred)
        rho = spearmanr(y_test_raw, pred).correlation

        fold_metrics.append({
            'fold': fold_idx, 'n_test_rows': len(test_idx),
            'held_out_groups': ','.join(map(str, held_out_groups[:5])) + ('...' if len(held_out_groups) > 5 else ''),
            'r2': r2, 'mae': mae, 'spearman_rho': rho,
        })
        recovered_note = ""
        if full_held_idx is not None and (len(full_held_idx) or len(zero_idx)):
            recovered_note = f", scored {len(full_held_idx):,} recovered low-WES bins"
            if len(zero_idx):
                recovered_note += f" ({len(zero_idx):,} hardcoded to 0.0, below threshold)"
        print(f"    -> Fold {fold_idx}/{n_splits}: n={len(test_idx):,}  R2={r2:.3f}  "
              f"MAE={mae:.3f}  rho={rho:.3f}{recovered_note}")

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(outdir / 'cv_metrics_per_fold.csv', index=False)

    out_predictions = df[['meta_sample', 'meta_chrom', 'meta_start', 'y_wgs_tumor_depth']].copy()
    out_predictions['oof_prediction'] = oof_pred
    out_predictions['residual'] = out_predictions['y_wgs_tumor_depth'] - out_predictions['oof_prediction']
    out_predictions.to_csv(outdir / 'cv_predictions.csv', index=False)

    if df_full is not None:
        out_recovered = df_full.loc[below_mask_full,
                                     ['meta_sample', 'meta_chrom', 'meta_start', 'y_wgs_tumor_depth']].copy()
        out_recovered['oof_prediction'] = oof_pred_full[below_mask_full]
        out_recovered['residual'] = out_recovered['y_wgs_tumor_depth'] - out_recovered['oof_prediction']
        if 'raw_wes_depth' in df_full.columns:
            out_recovered['raw_wes_depth'] = df_full.loc[below_mask_full, 'raw_wes_depth'].values
        n_unscored = int(out_recovered['oof_prediction'].isna().sum())
        if n_unscored:
            print(f"    [-] Warning: {n_unscored:,}/{len(out_recovered):,} recovered bins never "
                  f"got scored -- their sample/chrom didn't match any fold's held-out groups as "
                  f"expected. Check --cv-group is the same value used to prepare this run's "
                  f"groups, and that full_wes_range_data.parquet came from the same "
                  f"prepare_wgs_depth_features.py run as training_data.parquet.")
        out_recovered_path = outdir / 'cv_predictions_recovered_low_wes_depth.csv'
        out_recovered.to_csv(out_recovered_path, index=False)
        print(f"[+] Saved: {out_recovered_path}  ({len(out_recovered):,} rows, "
              f"{len(out_recovered) - n_unscored:,} scored) -- extrapolated below "
              f"--min-wes-depth, treat as lower-confidence (see module docstring)")

    overall_r2 = r2_score(y_raw, oof_pred)
    overall_mae = mean_absolute_error(y_raw, oof_pred)
    overall_rho = spearmanr(y_raw, oof_pred).correlation

    per_sample_rows = []
    for s, sub in out_predictions.groupby('meta_sample', observed=True):
        valid = sub['oof_prediction'].notna()
        if valid.sum() < 2:
            continue
        per_sample_rows.append({
            'sample': s, 'n_bins': int(valid.sum()),
            'r2': r2_score(sub.loc[valid, 'y_wgs_tumor_depth'], sub.loc[valid, 'oof_prediction']),
            'mae': mean_absolute_error(sub.loc[valid, 'y_wgs_tumor_depth'], sub.loc[valid, 'oof_prediction']),
            'spearman_rho': spearmanr(sub.loc[valid, 'y_wgs_tumor_depth'], sub.loc[valid, 'oof_prediction']).correlation,
        })
    pd.DataFrame(per_sample_rows).sort_values('r2').to_csv(outdir / 'per_sample_metrics.csv', index=False)

    plot_predicted_vs_actual(
        y_raw, oof_pred, outdir / 'predicted_vs_actual.png',
        title=f'Out-of-fold predictions ({n_splits}-fold, grouped by {args.cv_group}, '
              f'objective={args.objective})',
    )

    print(f"[+] Overall out-of-fold: R2={overall_r2:.3f}  MAE={overall_mae:.3f}  rho={overall_rho:.3f}")

    print("[+] Fitting final model on all data...")
    final_model = make_model(args.model, args.objective, args.huber_alpha, args.quantile_alpha, args.n_jobs)
    if args.model == 'lgbm':
        final_model.fit(X, y, categorical_feature=categorical_cols or 'auto')
        joblib.dump({'model': final_model, 'feature_cols': feature_cols,
                     'categorical_cols': categorical_cols, 'log_target': args.log_target,
                     'location_encoding': metadata['location_encoding'],
                     'objective': args.objective}, outdir / 'model.joblib')
        plot_feature_importance(final_model, feature_cols, outdir / 'feature_importance.png')
        pd.DataFrame({'feature': feature_cols, 'importance': final_model.feature_importances_}) \
            .sort_values('importance', ascending=False) \
            .to_csv(outdir / 'feature_importance.csv', index=False)
    else:
        X_rf = one_hot_for_rf(X, categorical_cols)
        final_model.fit(X_rf, y)
        joblib.dump({'model': final_model, 'feature_cols': list(X_rf.columns),
                     'categorical_cols': categorical_cols, 'log_target': args.log_target,
                     'location_encoding': metadata['location_encoding'],
                     'objective': args.objective}, outdir / 'model.joblib')
        plot_feature_importance(final_model, list(X_rf.columns), outdir / 'feature_importance.png')
        pd.DataFrame({'feature': X_rf.columns, 'importance': final_model.feature_importances_}) \
            .sort_values('importance', ascending=False) \
            .to_csv(outdir / 'feature_importance.csv', index=False)

    summary_path = outdir / 'cv_metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("WGS tumor depth model -- cross-validation summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Data dir:            {args.data_dir}\n")
        f.write(f"Model:               {args.model}\n")
        f.write(f"Objective:           {args.objective}"
                + (f" (alpha={args.huber_alpha})" if args.objective == 'huber' else '')
                + (f" (alpha={args.quantile_alpha})" if args.objective == 'quantile' else '')
                + "\n")
        f.write(f"Location encoding:   {metadata['location_encoding']}\n")
        f.write(f"Window size:         "
                + (f"{metadata.get('window_size', 0):,} bp (aggregated)"
                   if metadata.get('window_size', 0) else "native (no aggregation)")
                + "\n")
        f.write(f"CV group:            {args.cv_group}  ({n_groups} groups, {n_splits} folds)\n")
        f.write(f"Samples used:        {metadata['n_samples']}\n")
        f.write(f"Total training rows: {len(df):,}\n")
        f.write(f"Log-target:          {args.log_target}\n\n")
        f.write(f"Overall out-of-fold R2:      {overall_r2:.4f}\n")
        f.write(f"Overall out-of-fold MAE:     {overall_mae:.4f}\n")
        f.write(f"Overall out-of-fold rho:     {overall_rho:.4f}\n\n")
        f.write("Per-fold metrics:\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n")

    print(f"[+] Saved: {outdir / 'model.joblib'}")
    print(f"[+] Saved: {outdir / 'cv_predictions.csv'}  (includes per-row 'residual' column)")
    print(f"[+] Saved: {outdir / 'cv_metrics_per_fold.csv'}")
    print(f"[+] Saved: {outdir / 'per_sample_metrics.csv'}")
    print(f"[+] Saved: {outdir / 'feature_importance.csv'} / .png")
    print(f"[+] Saved: {outdir / 'predicted_vs_actual.png'}")
    print(f"[+] Saved: {summary_path}")
    print("\n[+] train_wgs_depth_model.py COMPLETE")


if __name__ == '__main__':
    main()
