"""Reproducible exploratory analysis; does not modify the simulator or source CSV.

Run: uv run --with pandas==3.0.6 --with scikit-learn==1.9.1
     --with matplotlib==3.11.2 python analysis/analyse_dataset.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'datasets/dump_truck_cwtvc_all_combinations.csv'
OUT = ROOT / 'analysis/results'
OUT.mkdir(parents=True, exist_ok=True)
EXCLUDE_PERCEPTION = False
PERCEPTION_PREFIXES = ('dss.', 'gnss.', 'lidar.', 'radar.')
TARGETS = ['engine_health_score', 'rul_hours']
LABELS = TARGETS + ['rul_low_hours', 'rul_high_hours', 'health_confidence',
                    'engine_health_classification', 'engine_health_traffic_light']
GROUP_KEYS = ['scenario_input', 'payload_class_input']
# Hold out both starts and every fault combination of each chosen base group.
TEST_GROUPS = {'ideal|heavy', 'stress|light', 'overhaul|medium'}
VAL_GROUPS = {'ideal|light', 'stress|medium', 'overhaul|heavy'}


def eligibility(name):
    if EXCLUDE_PERCEPTION and (name.lower().startswith(PERCEPTION_PREFIXES) or 'camera' in name.lower()):
        return 'excluded', 'Excluded by requested no GNSS/GPS, LiDAR, radar, DSS or camera analysis'
    if name in LABELS or name.startswith('engine_health.') or name in {
        'health_confidence_pct', 'recommended_action', 'active_fault_count',
        'faults_json', 'operational_statuses_json'} or name.startswith('operational_statuses.'):
        return 'excluded', 'Target, derived label, or post-diagnostic output (leakage)'
    if name.startswith('fault'):
        return 'excluded', 'Simulator injection flag or rule-derived fault, not a raw measurement'
    if name in {'payload_utilization_pct', 'body_position_pct', 'hoist_angle_deg',
                'chassis_pitch_deg', 'rack_deg', 'coolant_condition_pct'}:
        return 'candidate', 'Observable/estimated telemetry; availability must be confirmed in vehicle'
    if name.startswith(('engine.', 'imu.', 'transmission.', 'brake.', 'suspension.',
                        'frame.', 'gnss.', 'lidar.', 'radar.', 'dss.')):
        excluded = {
            'engine.power_kw', 'engine.effective_power_kw', 'engine.coolant_energy_flow_kw',
            'engine.fuel_energy_input_kw', 'engine.exhaust_energy_flow_kw',
            'engine.mechanical_loss_kw', 'engine.remainder_loss_kw', 'engine.coolant_energy_pct',
            'engine.exhaust_energy_pct', 'engine.transient_index_pct', 'frame.fatigue_index_pct',
            'gnss.latitude_deg', 'gnss.longitude_deg', 'gnss.altitude_m',
            'dss.driver_message', 'dss.alert_level', 'dss.collision_warning',
            'dss.rollover_warning', 'dss.overspeed_warning', 'dss.emergency_braking',
            'dss.fatigue_alert', 'transmission.status', 'gnss.fix_type', 'lidar.obstacle_class',
            'radar.target_class'}
        if name in excluded:
            return 'excluded', 'Derived energy/stress/diagnostic output or route-specific location/category'
        return 'candidate', 'Observable/estimated telemetry; some channels are synthetic proxies'
    return 'excluded', 'ID, time/cycle shortcut, simulator configuration, or redundant payload representation'


def heatmap(corr, path, title, annotate=False):
    size = len(corr)
    fig, ax = plt.subplots(figsize=(max(9, size*.40), max(7, size*.36)))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(size), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(size), corr.index, fontsize=7)
    ax.set_title(title)
    if annotate:
        for i in range(size):
            for j in range(size):
                ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if abs(corr.iloc[i,j]) > .65 else 'black')
    fig.colorbar(im, ax=ax, shrink=.6, label='Pearson r')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def confidence(score):
    return np.where(score >= 75, 'High', np.where(score >= 60, 'Medium', 'Low'))


def evaluate(y, prediction, name, k, split):
    pred = np.column_stack([np.clip(prediction[:,0], 0, 100), np.clip(prediction[:,1], 0, 12000)])
    return dict(method=name, attributes=k, split=split,
                health_mae=float(mean_absolute_error(y[:,0], pred[:,0])),
                health_rmse=float(np.sqrt(mean_squared_error(y[:,0], pred[:,0]))),
                health_r2=float(r2_score(y[:,0], pred[:,0])),
                rul_mae_hours=float(mean_absolute_error(y[:,1], pred[:,1])),
                rul_r2=float(r2_score(y[:,1], pred[:,1])),
                confidence_macro_f1=float(f1_score(confidence(y[:,0]), confidence(pred[:,0]),
                                                  average='macro', zero_division=0)))


def main():
    global OUT, EXCLUDE_PERCEPTION
    parser = argparse.ArgumentParser(description='Leakage-aware correlation and dimensionality analysis')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/results')
    parser.add_argument('--exclude-perception', action='store_true', help='Exclude GNSS/GPS, LiDAR, radar, DSS and camera-related inputs')
    args = parser.parse_args()
    OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    EXCLUDE_PERCEPTION = args.exclude_perception
    print('Loading source dataset', flush=True)
    df = pd.read_csv(SOURCE, low_memory=False)
    groups = df[GROUP_KEYS].agg('|'.join, axis=1)
    splits = pd.Series(np.where(groups.isin(TEST_GROUPS), 'test',
                       np.where(groups.isin(VAL_GROUPS), 'validation', 'train')), index=df.index)
    train, validation, test = [df.index[splits.eq(s)].to_numpy() for s in ('train','validation','test')]
    audit, candidates = [], []
    for name in df.columns:
        role, reason = eligibility(name)
        if role == 'candidate' and not (pd.api.types.is_numeric_dtype(df[name]) or
                                         pd.api.types.is_bool_dtype(df[name])):
            role, reason = 'excluded', 'Non-numeric; excluded from numeric sensor analysis'
        if role == 'candidate' and df.loc[train, name].nunique() <= 1:
            role, reason = 'excluded', 'Constant in training split'
        if role == 'candidate':
            candidates.append(name)
        audit.append(dict(column=name, role=role, reason=reason,
                          dtype=str(df[name].dtype), missing=int(df[name].isna().sum()),
                          unique=int(df[name].nunique())))
    X = df[candidates].astype(float)
    assert np.isfinite(X.to_numpy()).all(), 'Nonfinite predictors require training-only imputation'
    y = df[TARGETS].to_numpy()
    assert np.isfinite(y).all()
    # Exact duplicates removed using train only; canonical columns retain input order.
    unique, duplicates = [], []
    for c in candidates:
        equivalent = next((k for k in unique if np.array_equal(X.loc[train,c], X.loc[train,k])), None)
        if equivalent:
            duplicates.append(dict(removed=c, kept=equivalent, reason='Exact duplicate on train'))
        else:
            unique.append(c)
    X = X[unique]
    print(f'{len(df)} rows, {len(candidates)} numeric candidates, {len(unique)} unique predictors', flush=True)
    pd.DataFrame(audit).to_csv(OUT / 'column_audit.csv', index=False)
    pd.DataFrame(duplicates, columns=['removed','kept','reason']).to_csv(OUT / 'exact_duplicates.csv', index=False)
    corr = X.loc[train].corr()
    corr.to_csv(OUT / 'correlation_pearson.csv')
    X.loc[train].corr(method='spearman').to_csv(OUT / 'correlation_spearman.csv')
    heatmap(corr, OUT / 'correlation_all.png', 'Training telemetry correlation — no labels or injection flags')
    # Targets are compared separately, never concatenated into PCA inputs.
    target_corr = pd.DataFrame(index=unique)
    for target in TARGETS:
        target_corr[target + '_pearson'] = X.loc[train].corrwith(df.loc[train,target])
        target_corr[target + '_spearman'] = X.loc[train].rank().corrwith(df.loc[train,target].rank())
    target_corr.sort_values('engine_health_score_pearson', key=abs, ascending=False).to_csv(OUT / 'target_correlations.csv')
    pairs = [dict(attribute_a=a, attribute_b=b, pearson_r=corr.loc[a,b])
             for i,a in enumerate(unique) for b in unique[i+1:] if abs(corr.loc[a,b]) >= .95]
    pd.DataFrame(pairs).sort_values('pearson_r', key=abs, ascending=False).to_csv(OUT / 'high_correlation_pairs.csv', index=False)
    split_table = df[['combination_id'] + GROUP_KEYS + ['start_condition_input','fault_combination']].copy()
    split_table['split'] = splits
    split_table.drop_duplicates().to_csv(OUT / 'journey_splits.csv', index=False)
    # Bounded, balanced fitting sample; all validation/test rows used for scoring.
    fit_idx = df.loc[train].groupby('combination_id', sort=True).sample(n=52, random_state=42).index.to_numpy()
    yscale = StandardScaler().fit(y[fit_idx])
    def fit(features):
        model = ExtraTreesRegressor(n_estimators=80, max_depth=18, min_samples_leaf=3,
                                    n_jobs=2, random_state=42)
        model.fit(features[fit_idx], yscale.transform(y[fit_idx]))
        return model
    raw = X.to_numpy(dtype=np.float32)
    full_model = fit(raw)
    ranks = pd.Series(full_model.feature_importances_, index=unique).sort_values(ascending=False)
    ranks.rename('joint_target_training_importance').to_csv(OUT / 'feature_importance.csv')
    # Training importance chooses a representative for each correlated pair.
    kept, removed = [], []
    for name in ranks.index:
        partner = next((k for k in kept if abs(corr.loc[name,k]) >= .95), None)
        if partner:
            removed.append(dict(removed=name, kept=partner, pearson_r=corr.loc[name,partner]))
        else:
            kept.append(name)
    pd.DataFrame(removed).to_csv(OUT / 'correlation_pruned.csv', index=False)
    print(f'Correlation pruning: {len(unique)} -> {len(kept)} inputs', flush=True)
    metrics = [evaluate(y[validation], yscale.inverse_transform(full_model.predict(raw[validation])),
                        'Full telemetry', len(unique), 'validation')]
    models, arrays = {}, {}
    for k in sorted(set([min(n,len(kept)) for n in [8,12,16,24,len(kept)]])):
        features = kept[:k]
        arr = X[features].to_numpy(dtype=np.float32)
        print(f'Fitting original-attribute subset: {k}', flush=True)
        model = fit(arr)
        metrics.append(evaluate(y[validation], yscale.inverse_transform(model.predict(arr[validation])),
                                f'Selected {k}', k, 'validation'))
        models[k], arrays[k] = model, arr
    # Choose smallest subset within 10% of best validation joint-normalised MAE.
    subset_metrics = metrics[1:]
    score = lambda m: m['health_mae']/100 + m['rul_mae_hours']/12000
    best = min(map(score, subset_metrics))
    winner = min(m['attributes'] for m in subset_metrics if score(m) <= 1.10*best)
    selected = kept[:winner]
    scaler = StandardScaler().fit(X.loc[train,kept])
    scaled = scaler.transform(X[kept])
    pca = PCA(n_components=.95, svd_solver='full').fit(scaled[train])
    pcs = pca.transform(scaled).astype(np.float32)
    print(f'PCA: {len(kept)} sensors -> {pca.n_components_} components at 95% variance', flush=True)
    pca_model = fit(pcs)
    metrics.append(evaluate(y[validation], yscale.inverse_transform(pca_model.predict(pcs[validation])),
                            'PCA 95%', pcs.shape[1], 'validation'))
    for name, model, arr in [('Full telemetry',full_model,raw),
                             (f'Selected {winner}',models[winner],arrays[winner]),
                             ('PCA 95%',pca_model,pcs)]:
        metrics.append(evaluate(y[test], yscale.inverse_transform(model.predict(arr[test])), name, arr.shape[1], 'test'))
    metrics.append(evaluate(y[test], np.tile(y[fit_idx].mean(axis=0),(len(test),1)), 'Training mean baseline',0,'test'))
    pd.DataFrame(metrics).to_csv(OUT / 'model_comparison.csv', index=False)
    finish(df, splits, X, corr, kept, selected, ranks, target_corr, scaler, pca, pcs,
           yscale, models[winner], metrics, audit, duplicates, fit_idx)


def finish(df, splits, X, corr, kept, selected, ranks, target_corr, scaler, pca, pcs,
           yscale, model, metrics, audit, duplicates, fit_idx):
    names = [f'PC{i+1:02d}' for i in range(pcs.shape[1])]
    pd.DataFrame(pca.components_.T, index=kept, columns=names).rename_axis('attribute').to_csv(OUT / 'pca_weights.csv')
    variance = pd.DataFrame({'component': names, 'explained_variance_ratio': pca.explained_variance_ratio_,
                            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(OUT / 'pca_variance.csv', index=False)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(range(1,len(names)+1),pca.explained_variance_ratio_, label='Individual component')
    ax.plot(range(1,len(names)+1),variance.cumulative_variance, 'o-', color='#d46727',label='Cumulative variance')
    ax.axhline(.95, color='grey',ls='--', label='95% target')
    ax.set(xlabel='Principal component', ylabel='Fraction of training variance',title='PCA on standardised, correlation-pruned telemetry')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'pca_scree.png',dpi=160); plt.close(fig)
    corr.loc[selected, selected].to_csv(OUT / 'correlation_selected.csv')
    heatmap(corr.loc[selected, selected], OUT / 'correlation_selected.png',
            'Selected original attributes — training correlations', annotate=True)
    fig, ax = plt.subplots(figsize=(11, max(5,len(selected)*.33)))
    plot = target_corr.loc[selected,['engine_health_score_pearson','rul_hours_pearson']].rename(
        columns={'engine_health_score_pearson':'Health score','rul_hours_pearson':'RUL'})
    plot.iloc[::-1].plot.barh(ax=ax); ax.set(xlim=(-1,1), xlabel='Pearson r (training only)',title='Selected inputs vs targets')
    ax.axvline(0,color='grey');fig.tight_layout();fig.savefig(OUT / 'selected_target_correlations.png',dpi=160);plt.close(fig)
    selected_table = pd.DataFrame({'attribute':selected,'training_importance':ranks[selected].values,
                                  'health_pearson':target_corr.loc[selected,'engine_health_score_pearson'].values,
                                  'rul_pearson':target_corr.loc[selected,'rul_hours_pearson'].values})
    selected_table.to_csv(OUT / 'selected_attributes.csv', index=False)
    # Explicit metadata prefix: never pass metadata/targets as training X.
    metadata = pd.DataFrame({'meta_combination_id':df.combination_id,
                             'meta_step':df.step,'meta_elapsed_s':df.elapsed_s,
                             'meta_split':splits,
                             'meta_group':df[GROUP_KEYS].agg('|'.join,axis=1)})
    targets = df[LABELS].rename(columns=lambda c:'target_'+c)
    pd.concat([metadata, X[selected], targets],axis=1).to_csv(OUT / 'reduced_attributes_dataset.csv',index=False,float_format='%.7g')
    pd.concat([metadata, pd.DataFrame(pcs,columns=names,index=df.index),targets],axis=1).to_csv(
        OUT / 'pca_dataset.csv',index=False,float_format='%.7g')
    schema = {'input_columns':selected, 'target_columns':list(targets.columns),
              'metadata_columns':list(metadata.columns), 'pca_input_columns':kept,
              'pca_output_columns':names, 'caution':'Do not use meta_* or target_* columns as predictors.'}
    (OUT / 'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    joblib.dump({'input_columns':kept,'scaler':scaler,'pca':pca},OUT / 'pca_transform.joblib')
    joblib.dump({'input_columns':selected,'model':model,'target_scaler':yscale,'targets':TARGETS},
                OUT / 'selected_model.joblib')
    with SOURCE.open('rb') as handle:
        sha=hashlib.file_digest(handle,'sha256').hexdigest()
    train_rows=int(splits.eq('train').sum())
    manifest={'source':str(SOURCE.relative_to(ROOT)), 'source_sha256':sha,
              'exclude_perception':EXCLUDE_PERCEPTION,
              'rows':len(df),'original_columns':len(df.columns), 'numeric_candidate_columns':sum(a['role']=='candidate' for a in audit),
              'unique_telemetry_inputs':len(X.columns),'correlation_pruned_inputs':len(kept),
              'selected_input_count':len(selected),'pca_components':int(pca.n_components_),
              'pca_variance_retained':float(sum(pca.explained_variance_ratio_)),
              'train_rows':train_rows,'fit_rows':len(fit_idx),'validation_rows':int(splits.eq('validation').sum()),
              'test_rows':int(splits.eq('test').sum()),'test_groups':sorted(TEST_GROUPS),
              'validation_groups':sorted(VAL_GROUPS),'selection_rule':'Smallest original subset within 10% of best validation health_MAE/100 + RUL_MAE/12000',
              'versions':{'pandas':pd.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def table(rows, keys):
        text='| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'
        for row in rows:
            text+='| '+' | '.join(f'{row[k]:.3f}' if isinstance(row[k],(float,np.floating)) else str(row[k]) for k in keys)+' |\n'
        return text
    metric_keys=['method','attributes','health_mae','health_r2','rul_mae_hours','rul_r2','confidence_macro_f1']
    report=f'''# Attribute reduction for the dump-truck simulator

## Scope and recommendation

Analysed **{len(df):,} rows**, 960 configurations, 171 samples each, from the source CSV (unchanged).
Perception inputs excluded: **{EXCLUDE_PERCEPTION}**. When true, all GNSS/GPS, LiDAR, radar, DSS and camera-related columns are excluded before correlation, selection and PCA.
The request originally said 171 cycles; this existing dataset contains **171 samples in one 1,800 s journey**
per combination, not 171 independently simulated journeys. This analysis does not expand it.

**202 CSV columns are not 202 independent sensors.** After excluding outputs, configuration shortcuts,
diagnostic flags and constants, there are {manifest['numeric_candidate_columns']} numeric candidates,
{len(X.columns)} after exact deduplication, and {len(kept)} after train-only |Pearson r| >= 0.95 pruning.

The validation-selected original-attribute set has **{len(selected)} inputs**, a reduction of
**{100*(1-len(selected)/len(X.columns)):.1f}%** relative to the {len(X.columns)} eligible nonduplicate inputs.
Use `reduced_attributes_dataset.csv` for an interpretable starting point. It contains metadata and targets
as well as inputs: load **only the input_columns in feature_schema.json as X**.
This is a recommendation for predicting **the simulator's health and RUL**, not a validated minimum
sensor set for physical trucks or for collision, braking, driver-safety or structural-failure detection.

## Recommended original attributes

{table(selected_table.to_dict('records'),['attribute','training_importance','health_pearson','rul_pearson'])}

Training importance is an ExtraTrees multi-output importance with both targets standardised, not
causal importance. Correlated alternatives may substitute for one another. Low single-variable
correlation does not imply a variable is unnecessary; fault interactions can be nonlinear.

![Selected correlations](correlation_selected.png)
![Target correlations](selected_target_correlations.png)

## Correlation outputs

- `correlation_pearson.csv`: full matrix for the {len(X.columns)} eligible nonduplicate inputs.
- `correlation_spearman.csv`: rank correlations for the same inputs (nonlinear monotonic checks).
- `correlation_selected.csv`: smaller matrix for recommended attributes.
- `target_correlations.csv`: individual inputs vs health and RUL, kept separate from feature matrices.
- `high_correlation_pairs.csv`: strongly related pairs; `correlation_pruned.csv`: removed representatives.
- `column_audit.csv`: disposition of every original column; `exact_duplicates.csv`: exact repetitions.

All reported correlations and fitted transforms use **training rows only**. Correlations reflect the
synthetic generator and repeated fault variants; they do not establish real-machine causality.

## PCA dimensionality reduction

StandardScaler (train mean/std) -> PCA (train only) retains **{pca.n_components_} components** and
**{100*sum(pca.explained_variance_ratio_):.2f}%** of variance from {len(kept)} pruned inputs.
Inputs have different units (kPa, degrees, g, mm/s), so scaling is necessary.
`pca_weights.csv` contains component projection coefficients, not Pearson correlations.
`pca_variance.csv` records individual/cumulative variance.

**PCA reduces model dimensions, not sensor requirements.** Collect all {len(kept)} PCA input channels
in schema order to transform a new frame. Prefer original-attribute selection when the aim is fewer
measurements. Retaining 95% of variance does not guarantee retaining 95% of health-prediction information.

![PCA variance](pca_scree.png)

## Leakage-aware validation

No row-wise random train/test split. Group = **scenario + payload class**; both starts, all 32 fault
variants and all temporal samples stay together. Nine groups train, three validate, three test.
This prevents fault-counterpart rows from the same base configuration appearing in different splits.
Training rows: {train_rows:,}; validation: {manifest['validation_rows']:,}; test: {manifest['test_rows']:,}.
ExtraTrees fits use a reproducible balanced {len(fit_idx):,}-row training subsample (52 rows/configuration);
correlations/scaling/PCA use all training rows. Scores use every row in the relevant held-out split.
80 trees, depth <=18, minimum leaf 3, seed 42; both targets standardised for comparable influence.

Validation chooses subset size from 8/12/16/24/all correlation-pruned inputs (capped at available count).
The smallest subset within 10% of the best joint normalised validation MAE is selected before testing.
PCA uses the same downstream learner. Baseline predicts the training-target means.

### Validation comparison

{table([m for m in metrics if m['split']=='validation'],metric_keys)}

### Untouched test comparison

Health MAE is in **0–100 score points**; RUL MAE is in **hours**. Higher R²/F1 is better; lower MAE is better.

{table([m for m in metrics if m['split']=='test'],metric_keys)}

Test groups: {', '.join(sorted(TEST_GROUPS))}.
Validation groups: {', '.join(sorted(VAL_GROUPS))}.

Confidence macro-F1 is computed by mapping predicted health to the existing High/Medium/Low rules;
confidence is **not an independently trained or calibrated probability**. The source percentage column
uses fixed values 90/75/55; these were assigned by the dataset generator, not estimated uncertainty.

## Limitations that matter for training

1. Health is generated by a hand-written penalty rule; RUL is an algebraic function of health plus
   scenario; confidence/classification are deterministic health bands. Good scores mean rule imitation,
   not verified failure prediction or remaining physical life.
2. Never feed health, RUL, confidence, their ranges/classes, actions, contribution factors, fault
   injection flags, active-fault count or operational-status labels into the prediction model.
3. A single seed and a single synthetic C-WTVC profile are present. Fifteen base groups are too few
   for a strong generalisation claim. The held-out scenarios use the same underlying simulator rules.
   Validate with new seeds, different routes, held-out fault mechanisms and ultimately actual failures.
4. Synthetic channels can be overly correlated. Oil-condition estimates, perception confidence and
   structural proxies require actual sensor/estimator support. Keep safety-critical sensors even if
   this health-target experiment ranks them low. Do not remove IMU/LiDAR/radar solely on these results.
5. The model currently uses coolant as a block-temperature proxy, plus separately synthesised vibration
   velocity in mm/s and acceleration in g. Those are not interchangeable physical measurements.
   Operational-band decisions and health-score decisions are separate rules and may disagree.
6. Selecting against two outputs cannot prove coverage for transmission, hydraulic or brake failure
   labels not independently tested here. This dataset also contains no independently observed failure times.

## Deliverables and reuse

- `reduced_attributes_dataset.csv`: all source rows, selected inputs, labelled targets and split metadata.
- `pca_dataset.csv`: same rows and targets, with fitted component scores.
- `feature_schema.json`: exact input/target/metadata lists.
- `pca_transform.joblib`: train-fitted scaler, PCA and required ordered columns.
- `selected_model.joblib`: demonstration model with target scaler and input list.
- `journey_splits.csv`, `model_comparison.csv`, `manifest.json`: reproducible split and evaluation evidence.

Joblib files are Python serialisations: load only these trusted locally generated files.
Full re-run from Engine:

```bash
uv run --with pandas==3.0.6 --with scikit-learn==1.9.1 --with matplotlib==3.11.2 python analysis/analyse_dataset.py
```
'''
    (OUT / 'REPORT.md').write_text(report)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print('Selected:',selected,flush=True)
    print('Saved report, matrices, charts and reduced datasets to', OUT,flush=True)


if __name__ == '__main__':
    main()
