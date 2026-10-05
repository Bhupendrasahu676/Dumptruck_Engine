"""Post-hoc threshold repair; weights are frozen, original results preserved.

This follows an observed coarse-grid failure. Threshold fitting uses only the
original tuning trucks; reported test outcomes are exploratory second-use,
not a newly untouched test. No model reselection is performed.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from analysis.train_fault_models import predict_bundle,thresholds_for,metrics


def refine(output):
    output=Path(output)
    old=joblib.load(output/'selected_model.joblib')
    manifest=json.loads((output/'manifest.json').read_text())
    schema=json.loads((output/'feature_schema.json').read_text())
    targets=schema['target_columns'];inputs=old['input_columns']
    frame=pd.read_csv(manifest['source_csv'],usecols=inputs+targets+['split','engine_id','episode_id','step'],
                      dtype={k:'float32' for k in inputs+targets})
    va=frame[frame.split.eq('validation')&~frame.engine_id.eq(manifest['calibration_truck'])]
    te=frame[frame.split.eq('test')]
    yva=va[targets].to_numpy(np.int8);yte=te[targets].to_numpy(np.int8)
    pva=predict_bundle(old,va,va.episode_id.to_numpy());pte=predict_bundle(old,te,te.episode_id.to_numpy())
    rows=[];fault_rows=[]
    for name,cap in [('exact_validation_max_F1',None),('normal_guard_1pct_per_fault',.01)]:
        threshold=thresholds_for(yva,pva,max_normal_fpr=cap)
        for partition,y,p in [('validation_tuning',yva,pva),('test_second_use_exploratory',yte,pte)]:
            summary,detail=metrics(y,p,threshold,targets,name,partition)
            rows.append(summary);fault_rows.extend(detail)
        bundle={**old,'thresholds':threshold,'threshold_policy':name,
                'test_evidence_status':'post_hoc_second_use_exploratory_not_new_untouched_test',
                'deployment_ready':False}
        joblib.dump(bundle,output/('threshold_refined_model.joblib' if cap is None else 'alarm_guarded_model.joblib'),compress=3)
        pd.DataFrame({'fault':targets,'threshold':threshold,'normal_FPR_cap':cap,'tuning_partition':'original_two_validation_trucks'}).to_csv(output/(name+'_thresholds.csv'),index=False)
        if cap is not None:
            predictions=te[['engine_id','episode_id','step']].reset_index(drop=True)
            for k,t in enumerate(targets):
                fault=t.split('.',1)[1]
                predictions['truth.'+fault]=yte[:,k]
                predictions['probability.'+fault]=pte[:,k]
                predictions['prediction.'+fault]=(pte[:,k]>=threshold[k]).astype(np.int8)
            predictions.to_csv(output/'alarm_guarded_test_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'threshold_refinement_comparison.csv',index=False)
    pd.DataFrame(fault_rows).to_csv(output/'threshold_refinement_per_fault.csv',index=False)
    (output/'THRESHOLD_REFINEMENT.json').write_text(json.dumps({'weights_unchanged':True,'original_artifacts_preserved':True,
        'reason':'Original 0.01-step threshold grid cannot resolve near-constant calibrated aging scores.',
        'selection_data':'Only two original validation-tuning trucks; no test thresholds fitted',
        'test_caveat':'Test already inspected; new test metrics are exploratory second-use. Obtain independent seeds/schedules/trucks before a deployment claim.',
        'normal_FPR_cap_semantics':'1% per label on normal validation rows; not a guarantee of 1% overall or field false alarms',
        'comparisons':rows},indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'analysis/results_enhanced_multilabel')
    print(json.dumps(refine(parser.parse_args().output),indent=2))
