import numpy as np
import pandas as pd
import pytest
pytest.importorskip("sklearn", reason="Use the isolated fault-training environment")
from sklearn.ensemble import ExtraTreesClassifier
from analysis.train_fault_models import thresholds_for,metrics,tree_probability,sigmoid_fit,sigmoid_apply,markdown_table


def test_binary_metrics_thresholds_and_calibration_shapes():
    y=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)
    p=np.array([[.05,.05],[.15,.8],[.75,.1],[.95,.9]])
    thresholds=thresholds_for(y,p)
    summary,rows=metrics(y,p,thresholds,['one','two'],'fixture','validation')
    assert summary['macro_AP']==1 and summary['macro_F1']==1
    assert sum(r['FP'] for r in rows)==0
    cal=sigmoid_fit(y,p);result=sigmoid_apply(cal,p)
    assert result.shape==y.shape and np.isfinite(result).all()
    assert ((result>=0)&(result<=1)).all()
    assert '| macro_AP |' in markdown_table(pd.DataFrame({'macro_AP':[1.0]}))


def test_constant_tree_output_handled_without_wrong_column():
    X=np.arange(16,dtype=float).reshape(8,2)
    y=np.c_[np.zeros(8,dtype=int),np.ones(8,dtype=int)]
    model=ExtraTreesClassifier(n_estimators=2,random_state=1).fit(X,y)
    p=tree_probability(model,X)
    assert np.array_equal(p,y)


def test_thresholds_resolve_narrow_calibrated_probabilities_and_alarm_guard():
    y=np.array([[0,0],[0,0],[1,0],[1,0]],dtype=np.int8)
    p=np.array([[.03901,.01],[.03902,.01],[.03903,.01],[.03904,.01]])
    t=thresholds_for(y,p,max_normal_fpr=.01)
    assert t[0]==.03903 and t[1]>1
    summary,_=metrics(y,p,t,['one','two'],'fixture','validation')
    assert summary['normal_false_alarm_rate']==0
