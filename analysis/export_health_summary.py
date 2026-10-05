"""Export conservative minimum and weighted sensor-health trends; never fit RUL."""
import argparse
import csv
import json
from pathlib import Path

WEIGHTS={"engine":.30,"cooling":.15,"transmission":.15,"hydraulics":.15,
         "brakes":.10,"structure":.10,"perception":.05}


def export(output):
    output=Path(output)
    destination=output/'health_weighted_assessment.csv'
    with (output/'health_assessment.csv').open(newline='') as source, destination.open('w',newline='') as target:
        reader=csv.DictReader(source)
        writer=csv.DictWriter(target,fieldnames=['engine_id','episode_id','step','health_minimum','health_weighted',
                                               'health_status','critical_subsystems','root_cause_confirmed','rul_hours','rul_status'])
        writer.writeheader()
        for row in reader:
            scores=json.loads(row['subsystem_health_json'])
            complete=all(isinstance(scores.get(key),(int,float)) for key in WEIGHTS)
            minimum=min(scores.values()) if complete else None
            weighted=sum(WEIGHTS[key]*scores[key] for key in WEIGHTS) if complete else None
            critical=[key for key,value in scores.items() if value is not None and value<40]
            writer.writerow({key:row[key] for key in ('engine_id','episode_id','step')}|
                            {'health_minimum':minimum,'health_weighted':round(weighted,2) if weighted is not None else None,
                             'health_status':'unknown' if not complete else 'critical_heuristic' if critical else 'trend_only_not_safety_clearance',
                             'critical_subsystems':json.dumps(critical),'root_cause_confirmed':False,'rul_hours':None,
                             'rul_status':'unavailable_not_validated'})
    (output/'health_policy.json').write_text(json.dumps({'weights':WEIGHTS,
        'sensor_basis':'Existing observation-only subsystem residual/persistence heuristics, not learned real condition labels',
        'aggregation':'Minimum for conservative flag, weighted for trend; all required subsystems must be observed',
        'DL_probabilities':'Kept separate; no arbitrary probability multiplier or double-counting of the same sensor deviation',
        'RUL':'No independent outcomes/censoring data: no physical RUL model trained and no point estimate fabricated'},indent=2)+'\n')
    return str(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('analysis/results_enhanced_multilabel'))
    print(export(parser.parse_args().output))
