import csv
import json
from analysis.export_health_summary import export


def test_missing_subsystem_stays_unknown_and_RUL_unavailable(tmp_path):
    keys=['engine_id','episode_id','step','subsystem_health_json']
    scores={'engine':90,'cooling':80,'transmission':70,'hydraulics':60,'brakes':50,'structure':40,'perception':30}
    with (tmp_path/'health_assessment.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerow({'engine_id':'v1','episode_id':'e1','step':0,'subsystem_health_json':json.dumps(scores)})
        writer.writerow({'engine_id':'v1','episode_id':'e1','step':1,'subsystem_health_json':json.dumps({**scores,'engine':None})})
    export(tmp_path)
    with (tmp_path/'health_weighted_assessment.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert float(rows[0]['health_minimum'])==30
    assert float(rows[0]['health_weighted'])==69
    assert rows[1]['health_weighted']=='' and rows[1]['health_status']=='unknown'
    assert all(row['rul_hours']=='' and row['root_cause_confirmed']=='False' for row in rows)
