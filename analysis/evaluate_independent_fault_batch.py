"""Fresh-seed evaluation of frozen weights/thresholds; never trains or tunes."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from generate_training_csv import generate_training_csv
from analysis.train_fault_models import metrics,predict_bundle


def evaluate(output,seed=20261006,model_filename="alarm_guarded_model.joblib"):
    output=Path(output)
    target=output/'independent_seed_validation'
    target.mkdir(parents=True,exist_ok=True)
    csv_path=target/'independent_fault_batch.csv'
    # Frozen policies must exist BEFORE data are created/evaluated.
    if Path(model_filename).name!=model_filename:
        raise ValueError("model_filename must name a bundle inside the results folder")
    bundle=joblib.load(output/model_filename)
    receipt=generate_training_csv(csv_path,engines=3,episodes_per_fault=1,samples_per_journey=171,seed=seed)
    generated=json.loads(csv_path.with_suffix('.manifest.json').read_text())
    original=json.loads((ROOT/'datasets/dump_truck_enhanced_all_conditions.manifest.json').read_text())
    new_seeds={e['config']['engine_seed'] for e in generated['episodes']}
    old_seeds={e['config']['engine_seed'] for e in original['episodes']}
    assert not new_seeds & old_seeds,'External virtual-engine variant overlaps original data'
    assert seed!=original['parameters']['seed']
    inputs,targets=bundle['input_columns'],bundle['target_columns']
    data=pd.read_csv(csv_path,usecols=inputs+targets+['engine_id','episode_id','step'],
                     dtype={key:'float32' for key in inputs+targets})
    y=data[targets].to_numpy(np.int8)
    p=predict_bundle(bundle,data,data.episode_id.to_numpy())
    summary,per_fault=metrics(y,p,bundle['thresholds'],targets,'frozen_alarm_guarded_model','fresh_seed_external_all_rows')
    pd.DataFrame(per_fault).to_csv(target/'per_fault_metrics.csv',index=False)
    results=data[['engine_id','episode_id','step']].copy()
    for k,name in enumerate(targets):
        fault=name.split('.',1)[1]
        results['truth.'+fault]=y[:,k]
        results['probability.'+fault]=p[:,k]
        results['prediction.'+fault]=(p[:,k]>=bundle['thresholds'][k]).astype(np.int8)
    results.to_csv(target/'predictions.csv',index=False)
    record={'score':summary,'seed':seed,'rows':len(data),'receipt':receipt,'model_filename':model_filename,
            'independent_engine_variant_seeds':sorted(new_seeds),'overlap_original_variant_seeds':False,
            'weights_and_thresholds_frozen_before_generation':True,'training_or_tuning_performed':False,
            'batch_semantics':'All rows are external evaluation, regardless locally generated split metadata. Engine ID strings are batch-local; independent engine_seed sets establish distinct synthetic variants.',
            'limitations':'Only3additional virtualvariants, identical physics/routefamily; not field validation or confirmed rootcauses/RUL'}
    (target/'RESULTS.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'analysis/results_enhanced_multilabel')
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--model-file',default='alarm_guarded_model.joblib')
    args=parser.parse_args();print(json.dumps(evaluate(args.output,args.seed,args.model_file),indent=2))
