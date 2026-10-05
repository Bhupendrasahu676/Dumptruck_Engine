"""Verify saved synthetic classification results; never change fitted models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.train_fault_models import metrics


def verify(output):
    output=Path(output)
    manifest=json.loads((output/"manifest.json").read_text())
    schema=json.loads((output/"feature_schema.json").read_text())
    table=pd.read_csv(output/"test_predictions.csv")
    targets=schema["target_columns"]
    faults=[t.split('.',1)[1] for t in targets]
    y=table[['truth.'+f for f in faults]].to_numpy(np.int8)
    p=table[['probability.'+f for f in faults]].to_numpy()
    predictions=table[['prediction.'+f for f in faults]].to_numpy(np.int8)
    thresholds=pd.read_csv(output/"thresholds.csv").set_index('fault').loc[targets,'threshold'].to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert np.array_equal(predictions,p>=thresholds)
    assert len(table)==manifest['test_rows']
    assert set(table.engine_id)==set(manifest['split_membership']['test'])
    assert not table.duplicated(['episode_id','step']).any()
    summary,rows=metrics(y,p,thresholds,targets,manifest['selected_model']+'_sigmoid_calibrated','test')
    comparisons=pd.read_csv(output/'model_comparison.csv')
    saved=comparisons[(comparisons.candidate.eq(summary['candidate']))&(comparisons.split.eq('test'))].iloc[0]
    for field in ('macro_AP','macro_F1','macro_recall','micro_F1','normal_false_alarm_rate'):
        assert np.isclose(saved[field],summary[field],atol=1e-7),(field,saved[field],summary[field])
    # Check original artifact snapshots. Later docs are intentionally not part
    # of the original run's digest inventory.
    for relative,digest in manifest['artifacts'].items():
        actual=hashlib.sha256((output/relative).read_bytes()).hexdigest()
        assert actual==digest,relative
    groups={engine:np.flatnonzero(table.engine_id.eq(engine).to_numpy()) for engine in table.engine_id.unique()}
    rng=np.random.default_rng(manifest['seed'])
    bootstrap=[]
    engines=list(groups)
    for _ in range(200):
        indices=np.concatenate([groups[engine] for engine in rng.choice(engines,len(engines),replace=True)])
        score,_=metrics(y[indices],p[indices],thresholds,targets,'bootstrap','test')
        bootstrap.append({key:score[key] for key in ('macro_AP','macro_F1','macro_recall','normal_false_alarm_rate')})
    intervals={key:{'low_2_5':float(np.quantile([v[key] for v in bootstrap],.025)),
                    'high_97_5':float(np.quantile([v[key] for v in bootstrap],.975))}
               for key in bootstrap[0]}
    per_fault=pd.DataFrame(rows)
    weakest=per_fault.sort_values('AP').head(5)
    receipt={'verified':True,'rows':len(table),'probability_columns':len(faults),
             'metrics_recomputed':summary,'artifact_digest_checks':len(manifest['artifacts']),
             'truck_bootstrap_200':intervals,
             'bootstrap_caveat':'Only three synthetic test trucks; intervals are fragile, not field confidence',
             'weakest_faults':weakest[['fault','AP','recall','F1']].to_dict('records')}
    (output/'RESULTS_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ax=per_fault.set_index('fault')['AP'].plot.barh(figsize=(10,7),xlim=(0,1),title='Selected model — held-truck synthetic per-fault AP')
        ax.figure.tight_layout();ax.figure.savefig(output/'per_fault_AP.png',dpi=150);plt.close(ax.figure)
    except ImportError:
        receipt['plot']='matplotlib unavailable; numeric CSV remains authoritative'
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/results_enhanced_multilabel')
    args=parser.parse_args()
    print(json.dumps(verify(args.output),indent=2))
