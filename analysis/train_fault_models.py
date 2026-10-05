"""Train-only feature selection and held-truck multilabel synthetic fault study."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                            hamming_loss, log_loss, brier_score_loss, precision_recall_curve)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from analysis.fault_features import select_features
from analysis import fault_dl as dl
from analysis.fault_feature_policy import filter_input_names, assert_allowed_names


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def markdown_table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        values=[f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value) for value in row]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def thresholds_for(y, p, max_normal_fpr=None):
    thresholds = []
    normal=y.sum(axis=1)==0
    for k in range(y.shape[1]):
        if not y[:,k].sum():
            thresholds.append(1.000001)
            continue
        precision,recall,grid=precision_recall_curve(y[:,k],p[:,k])
        scores=2*precision[:-1]*recall[:-1]/np.maximum(1e-12,precision[:-1]+recall[:-1])
        if max_normal_fpr is not None and normal.any():
            negative=np.sort(p[normal,k])
            fpr=(len(negative)-np.searchsorted(negative,grid,side='left'))/len(negative)
            scores=np.where(fpr<=max_normal_fpr,scores,-1.)
        thresholds.append(float(grid[np.argmax(scores)]) if (scores>=0).any() else 1.000001)
    return np.asarray(thresholds)


def metrics(y, p, thresholds, names, candidate, partition):
    pred = p >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    ap = np.array([average_precision_score(y[:, k], p[:, k]) if y[:, k].sum() else np.nan for k in range(y.shape[1])])
    normal = y.sum(axis=1) == 0
    rows = []
    for k, name in enumerate(names):
        true, detected = y[:, k].astype(bool), pred[:, k]
        rows.append({"candidate": candidate, "split": partition, "fault": name, "AP": ap[k],
                     "precision": precision[k], "recall": recall[k], "F1": f1[k],
                     "support": int(support[k]), "threshold": thresholds[k],
                     "TP": int((true & detected).sum()), "FP": int((~true & detected).sum()),
                     "FN": int((true & ~detected).sum()), "TN": int((~true & ~detected).sum()),
                     "Brier": brier_score_loss(y[:, k], p[:, k])})
    summary = {"candidate": candidate, "split": partition, "macro_AP": float(np.nanmean(ap)) if np.isfinite(ap).any() else None,
               "micro_AP": average_precision_score(y.ravel(), p.ravel()) if y.sum() else None,
               "macro_F1": float(f1.mean()), "macro_recall": float(recall.mean()),
               "micro_F1": float(precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[2]),
               "hamming_loss": hamming_loss(y, pred), "exact_match": float((y == pred).all(axis=1).mean()),
               "normal_false_alarm_rate": float(pred[normal].any(axis=1).mean()) if normal.any() else None,
               "macro_Brier": float(np.mean([r["Brier"] for r in rows])), "rows": len(y)}
    return summary, rows


def tree_probability(model, X):
    outputs = model.predict_proba(X)
    return np.column_stack([v[:, list(classes).index(1)] if 1 in classes else np.zeros(len(X))
                            for v, classes in zip(outputs, model.classes_)])


def sigmoid_fit(y, p):
    calibrators = []
    for k in range(y.shape[1]):
        logits = np.log(np.clip(p[:, k], 1e-6, 1-1e-6) / np.clip(1-p[:, k], 1e-6, 1))[:, None]
        if len(np.unique(y[:, k])) < 2:
            calibrators.append(None)
        else:
            calibrators.append(LogisticRegression(C=1., max_iter=200).fit(logits, y[:, k]))
    return calibrators


def sigmoid_apply(calibrators, p):
    result = p.copy()
    for k, model in enumerate(calibrators):
        if model is not None:
            logit = np.log(np.clip(p[:, k], 1e-6, 1-1e-6) / np.clip(1-p[:, k], 1e-6, 1))[:, None]
            result[:, k] = model.predict_proba(logit)[:, 1]
    return result


def predict_bundle(bundle, frame, episode_ids=None):
    X = frame[bundle["input_columns"]].to_numpy(dtype=np.float32)
    Z = bundle["imputer"].transform(frame[bundle["input_columns"]])
    if bundle.get("scaler") is not None:
        Z = bundle["scaler"].transform(Z).astype(np.float32)
    if bundle.get("pca") is not None:
        Z = bundle["pca"].transform(Z).astype(np.float32)
    if bundle.get("autoencoder_path"):
        Z = dl.encode(dl.load_model(bundle["autoencoder_path"]), Z)
    if bundle["kind"] == "tree":
        p = tree_probability(bundle["model"], Z)
    elif bundle["kind"] == "logistic":
        p = bundle["model"].predict_proba(Z)
    else:
        loaded = dl.load_model(bundle["model_path"])
        p = dl.predict_tcn(loaded, Z, episode_ids) if bundle["kind"] == "tcn" else dl.predict_dense(loaded, Z)
    if bundle.get("calibrators"):
        p = sigmoid_apply(bundle["calibrators"], p)
    return p


def run(csv_path, output, epochs=18, seed=793, exclude_gnss_brakes=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new results folder; existing artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    schema = json.loads(Path(csv_path).with_suffix(".schema.json").read_text())
    manifest = json.loads(Path(csv_path).with_suffix(".manifest.json").read_text())
    names, removed_inputs = filter_input_names(schema["input_columns"], exclude_gnss_brakes=exclude_gnss_brakes)
    targets = schema["target_columns"]
    if exclude_gnss_brakes:
        assert_allowed_names(names)
    pd.DataFrame(removed_inputs, columns=["attribute", "reason"]).to_csv(output/"excluded_attributes.csv",index=False)
    exclusion_policy={"exclude_gnss_brakes":exclude_gnss_brakes,"original_input_count":len(schema["input_columns"]),
                      "remaining_input_count":len(names),"removed_inputs":removed_inputs,
                      "target_policy":"All16 labels retained for comparison; brake_wear lacks direct evidence and is not approved for prediction" if exclude_gnss_brakes else "All16 labels retained"}
    (output/"input_exclusion_policy.json").write_text(json.dumps(exclusion_policy,indent=2))
    metadata = ["engine_id", "episode_id", "step", "elapsed_s", "split", "primary_class", "sensor_quality_json", "overall_health", "subsystem_health_json", "root_cause_confirmed", "rul_status"]
    severity = ["effect_level." + n.split(".", 1)[1] for n in targets]
    print("Loading approved CSV columns", flush=True)
    frame = pd.read_csv(csv_path, usecols=names+targets+metadata+severity,
                        dtype={n: "float32" for n in names+targets+severity})
    for engine, rows in frame.groupby("engine_id"):
        if rows["split"].nunique() != 1:
            raise ValueError("Truck crosses partitions")
        if engine not in manifest["split_membership"][rows["split"].iloc[0]]:
            raise ValueError("Split membership mismatch")
    if frame.duplicated(["episode_id", "step"]).any():
        raise ValueError("Duplicate episode steps")
    tr = frame[frame.split.eq("train")].copy()
    validation = frame[frame.split.eq("validation")].copy()
    calibration_truck = sorted(validation.engine_id.unique())[0]
    ca = validation[validation.engine_id.eq(calibration_truck)].copy()
    va = validation[~validation.engine_id.eq(calibration_truck)].copy()
    te = frame[frame.split.eq("test")].copy()
    Xtr, ytr = tr[names].to_numpy(np.float32), tr[targets].to_numpy(np.int8)
    yva, yca, yte = [f[targets].to_numpy(np.int8) for f in (va, ca, te)]
    print("Training-only selection", flush=True)
    selection = select_features(Xtr, ytr, names, targets, output / "feature_selection", seed=seed)
    pd.DataFrame([{ "episode_id": ep, "engine_id": group.engine_id.iloc[0], "split": group.split.iloc[0],
                   "rows": len(group)} for ep, group in frame.groupby("episode_id", sort=False)]).to_csv(output/"journey_splits.csv", index=False)
    candidates, comparison, per_fault = {}, [], []
    def matrices(features):
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        a = imp.fit_transform(tr[features]).astype(np.float32)
        return imp, a, *(imp.transform(f[features]).astype(np.float32) for f in (va, ca, te))
    def record(name, bundle, pva, pca, pte):
        thresholds = thresholds_for(yva, pva)
        summary, rows = metrics(yva, pva, thresholds, targets, name, "validation_tuning")
        summary["inputs"] = len(bundle["input_columns"])
        comparison.append(summary); per_fault.extend(rows)
        candidates[name] = {"bundle": bundle, "pva": pva, "pca": pca, "pte": pte,
                            "thresholds": thresholds, "summary": summary}
        print(name, {k:round(summary[k],4) for k in ("macro_AP","macro_F1","macro_recall")}, flush=True)
    rng = np.random.default_rng(seed)
    fit_indices = np.sort(rng.choice(len(tr), min(40000, len(tr)), replace=False))
    for key in ("all", "correlation_pruned", "64", "32", "16", "8", "proxy_ablated"):
        features = selection["subsets"][key]
        imp, a, b, c, d = matrices(features)
        model = ExtraTreesClassifier(n_estimators=80, max_depth=18, max_leaf_nodes=2048,
                                     min_samples_leaf=3, n_jobs=2, random_state=seed)
        model.fit(a[fit_indices], ytr[fit_indices])
        record("tree_"+key, {"kind":"tree", "model":model, "imputer":imp, "input_columns":features},
               tree_probability(model,b), tree_probability(model,c), tree_probability(model,d))
    # Validation-only subset choice; do not inspect test scores here.
    best = max(candidates, key=lambda n:candidates[n]["summary"]["macro_AP"])
    best_ap = candidates[best]["summary"]["macro_AP"]
    acceptable = [n for n in candidates if candidates[n]["summary"]["macro_AP"] >= best_ap-.015]
    selected_tree = min(acceptable, key=lambda n: candidates[n]["summary"]["inputs"])
    features = candidates[selected_tree]["bundle"]["input_columns"]
    (output/"selected_attributes.csv").write_text(pd.DataFrame({"attribute":features}).to_csv(index=False))
    print("Selected original subset",selected_tree,flush=True)
    # Refit original subset tree on every training row.
    imp,a,b,c,d = matrices(features)
    full_tree = ExtraTreesClassifier(n_estimators=120,max_depth=20,max_leaf_nodes=3072,min_samples_leaf=3,n_jobs=2,random_state=seed)
    full_tree.fit(a,ytr)
    record("tree_selected_full_fit", {"kind":"tree","model":full_tree,"imputer":imp,"input_columns":features},
           tree_probability(full_tree,b),tree_probability(full_tree,c),tree_probability(full_tree,d))
    scale=StandardScaler();a=scale.fit_transform(a).astype(np.float32)
    b,c,d=[scale.transform(z).astype(np.float32) for z in (b,c,d)]
    logistic=OneVsRestClassifier(LogisticRegression(max_iter=250,class_weight="balanced",C=1),n_jobs=2)
    logistic.fit(a,ytr)
    record("logistic_selected",{"kind":"logistic","model":logistic,"imputer":imp,"scaler":scale,"input_columns":features},
           logistic.predict_proba(b),logistic.predict_proba(c),logistic.predict_proba(d))
    dense=dl.train_dense(a,ytr,b,yva,output/"models",name="dense_selected",seed=seed,epochs=epochs)
    record("dense_selected",{"kind":"dense","model_path":dense["model_path"],"imputer":imp,"scaler":scale,"input_columns":features},
           dl.predict_dense(dense,b),dl.predict_dense(dense,c),dl.predict_dense(dense,d))
    tcn=dl.train_tcn(a,ytr,tr.episode_id.to_numpy(),b,yva,va.episode_id.to_numpy(),output/"models",seed=seed,epochs=epochs,window=12)
    record("tcn_selected",{"kind":"tcn","model_path":tcn["model_path"],"imputer":imp,"scaler":scale,"input_columns":features,"window":12},
           dl.predict_tcn(tcn,b,va.episode_id.to_numpy()),dl.predict_tcn(tcn,c,ca.episode_id.to_numpy()),dl.predict_tcn(tcn,d,te.episode_id.to_numpy()))
    full_features=selection["eligible_names"]
    imp_full, af,bf,cf,df = matrices(full_features)
    scaler_full=StandardScaler();af=scaler_full.fit_transform(af).astype(np.float32)
    bf,cf,df=[scaler_full.transform(z).astype(np.float32) for z in (bf,cf,df)]
    for retained in (.90,.95,.99):
        pca=PCA(n_components=retained,svd_solver="full",random_state=seed)
        ap=pca.fit_transform(af).astype(np.float32)
        bp,cp,dp=[pca.transform(z).astype(np.float32) for z in (bf,cf,df)]
        name=f"dense_pca_{int(retained*100)}"
        model=dl.train_dense(ap,ytr,bp,yva,output/"models",name=name,seed=seed,epochs=epochs)
        record(name,{"kind":"dense","model_path":model["model_path"],"imputer":imp_full,"scaler":scaler_full,"pca":pca,"input_columns":full_features,"components":pca.n_components_},
               dl.predict_dense(model,bp),dl.predict_dense(model,cp),dl.predict_dense(model,dp))
        pd.DataFrame(pca.components_.T,index=full_features).to_csv(output/f"{name}_loadings.csv")
    ae=dl.train_autoencoder(af,bf,output/"models",seed=seed,latent_dim=16,epochs=min(epochs,15))
    ae_a,ae_b,ae_c,ae_d=[dl.encode(ae,z) for z in (af,bf,cf,df)]
    ae_head=dl.train_dense(ae_a,ytr,ae_b,yva,output/"models",name="dense_autoencoder",seed=seed,epochs=epochs)
    record("dense_autoencoder",{"kind":"dense","model_path":ae_head["model_path"],"autoencoder_path":ae["model_path"],"imputer":imp_full,"scaler":scaler_full,"input_columns":full_features,"components":16},
           dl.predict_dense(ae_head,ae_b),dl.predict_dense(ae_head,ae_c),dl.predict_dense(ae_head,ae_d))
    prior=np.tile(ytr.mean(axis=0),(len(va),1))
    baseline,_=metrics(yva,prior,thresholds_for(yva,prior),targets,"training_prevalence_baseline","validation_tuning")
    baseline["inputs"]=0;comparison.append(baseline)
    best_name=max(candidates,key=lambda n:candidates[n]["summary"]["macro_AP"])
    best_dl=max((n for n in candidates if candidates[n]["bundle"]["kind"] in ("dense","tcn")),key=lambda n:candidates[n]["summary"]["macro_AP"])
    print("Frozen model selection:",best_name,"DL:",best_dl,flush=True)
    # Reserve one validation truck for calibration; threshold/model tuning
    # uses the other two. Test outcomes never select or calibrate a model.
    calibrated={}
    calibration_metrics=[]
    calibration_bins=[]
    for name in dict.fromkeys((best_name,best_dl)):
        candidate=candidates[name]
        calibrators=sigmoid_fit(yca,candidate["pca"])
        pva=sigmoid_apply(calibrators,candidate["pva"])
        pte=sigmoid_apply(calibrators,candidate["pte"])
        thresholds=thresholds_for(yva,pva,max_normal_fpr=.01 if exclude_gnss_brakes else None)
        bundle={**candidate["bundle"],"calibrators":calibrators,"thresholds":thresholds,
                "target_columns":targets,"synthetic_only":True,"root_cause_confirmed":False,
                "rul_status":"unavailable_not_validated","deployment_ready":False,
                "input_exclusion_policy":exclusion_policy}
        bundle["threshold_policy"]="exact_validation_max_F1_with_1pct_per_label_normal_guard" if exclude_gnss_brakes else "exact_validation_max_F1"
        bundle["unsupported_targets"]=["fault_label.brake_wear"] if exclude_gnss_brakes else []
        if exclude_gnss_brakes:
            assert_allowed_names(bundle["input_columns"])
        path=output/("selected_model.joblib" if name==best_name else "selected_dl_model.joblib")
        joblib.dump(bundle,path,compress=3)
        calibrated[name]={"bundle":bundle,"pva":pva,"pte":pte,"thresholds":thresholds,"path":str(path)}
        for partition,y,p0,p1 in (("validation_tuning",yva,candidate["pva"],pva),("test",yte,candidate["pte"],pte)):
            for state,prob in (("raw",p0),("sigmoid_calibrated",p1)):
                summary,rows=metrics(y,prob,thresholds if state!="raw" else candidate["thresholds"],targets,name+"_"+state,partition)
                summary["inputs"]=len(bundle["input_columns"]);comparison.append(summary);per_fault.extend(rows)
                for k,target in enumerate(targets):
                    calibration_metrics.append({"candidate":name,"split":partition,"state":state,"fault":target,
                                                "Brier":brier_score_loss(y[:,k],prob[:,k]),"log_loss":log_loss(y[:,k],prob[:,k],labels=[0,1])})
                    for lo in np.linspace(0,.9,10):
                        mask=(prob[:,k]>=lo)&(prob[:,k]<(lo+.1) if lo<.9 else prob[:,k]<=1)
                        calibration_bins.append({"candidate":name,"split":partition,"state":state,"fault":target,"bin_low":lo,
                                                 "count":int(mask.sum()),"mean_probability":float(prob[mask,k].mean()) if mask.any() else None,
                                                 "observed_frequency":float(y[mask,k].mean()) if mask.any() else None})
        # Serialization proof on a full episode prefix, including TCN boundaries.
        reloaded=joblib.load(path)
        check=predict_bundle(reloaded,te.iloc[:171],te.episode_id.to_numpy()[:171])
        np.testing.assert_allclose(check,pte[:171],rtol=1e-5,atol=1e-6)
    # Freeze all candidates before test comparison; report every predeclared one.
    for name,candidate in candidates.items():
        summary,rows=metrics(yte,candidate["pte"],candidate["thresholds"],targets,name,"test")
        summary["inputs"]=len(candidate["bundle"]["input_columns"]);comparison.append(summary);per_fault.extend(rows)
    selected=calibrated[best_name];p=selected["pte"];thresholds=selected["thresholds"]
    pred=p>=thresholds
    prediction=te[["engine_id","episode_id","step","elapsed_s"]].reset_index(drop=True)
    for k,target in enumerate(targets):
        fault=target.split(".",1)[1]
        prediction["truth."+fault]=yte[:,k]
        prediction["probability."+fault]=p[:,k]
        prediction["prediction."+fault]=pred[:,k].astype(int)
    prediction.to_csv(output/"test_predictions.csv",index=False)
    pd.DataFrame(comparison).to_csv(output/"model_comparison.csv",index=False)
    pd.DataFrame(per_fault).to_csv(output/"per_fault_metrics.csv",index=False)
    pd.DataFrame(calibration_metrics).to_csv(output/"calibration_metrics.csv",index=False)
    pd.DataFrame(calibration_bins).to_csv(output/"calibration_bins.csv",index=False)
    pd.DataFrame({"fault":targets,"threshold":thresholds,"objective":"max_F1_with_1pct_per_label_normal_guard" if exclude_gnss_brakes else "max_F1","tuning_partition":"two_validation_trucks"}).to_csv(output/"thresholds.csv",index=False)
    stratified=[]
    strength=te[severity].to_numpy()
    strata={"normal":yte.sum(axis=1)==0,"single_fault":yte.sum(axis=1)==1,"mixed_fault":yte.sum(axis=1)>1,
            "sensor_missing":te.sensor_quality_json.str.contains('"missing"').to_numpy(),
            "any_weak_effect":((strength> .001)&(strength<.5)).any(axis=1),
            "any_developed_effect":(strength>=.5).any(axis=1)}
    strata.update({"truck_"+engine:te.engine_id.eq(engine).to_numpy() for engine in te.engine_id.unique()})
    for stratum,mask in strata.items():
        if mask.any():
            summary,_=metrics(yte[mask],p[mask],thresholds,targets,best_name,stratum)
            stratified.append(summary)
    pd.DataFrame(stratified).to_csv(output/"stratified_metrics.csv",index=False)
    # Episode-level first detection; label truth only defines evaluation segments.
    delays=[]
    for ep,positions in te.reset_index(drop=True).groupby("episode_id",sort=False).indices.items():
        positions=np.asarray(positions)
        times=te.iloc[positions].elapsed_s.to_numpy()
        for k,target in enumerate(targets):
            true=yte[positions,k].astype(bool)
            starts=np.flatnonzero(true & ~np.r_[False,true[:-1]])
            for start in starts:
                later=np.flatnonzero(~true[start:])
                end=start+later[0] if len(later) else len(true)
                hits=np.flatnonzero(pred[positions[start:end],k])
                delays.append({"episode_id":ep,"fault":target,"onset_s":times[start],"detected":bool(len(hits)),
                               "delay_s":float(times[start+hits[0]]-times[start]) if len(hits) else None})
    pd.DataFrame(delays).to_csv(output/"detection_delay.csv",index=False)
    # Missing-sensor robustness: observation-only masked/imputed selected columns.
    robust=te.iloc[:min(8192,len(te))].copy()
    raw=[n for n in selected["bundle"]["input_columns"] if "__" not in n and not n.startswith(("context.","leadlag."))]
    for name in raw[::5]:
        robust[name]=np.nan
        if name+"__available_mask" in robust:robust[name+"__available_mask"]=0
        for col in selected["bundle"]["input_columns"]:
            if col.startswith(name+"__") and not col.endswith("_mask"):robust[col]=np.nan
    rp=predict_bundle(selected["bundle"],robust,robust.episode_id.to_numpy())
    robustness,_=metrics(yte[:len(robust)],rp,thresholds,targets,best_name,"stress_mask_every_fifth_selected_raw_channel")
    (output/"missing_sensor_robustness.json").write_text(json.dumps(robustness,indent=2))
    # Explainability: permutation on validation, never test feature selection.
    feature_names=selected["bundle"]["input_columns"]
    sample=va.iloc[:4096].copy();sample_y=yva[:len(sample)]
    base=predict_bundle(selected["bundle"],sample,sample.episode_id.to_numpy())
    base_ap=metrics(sample_y,base,thresholds,targets,best_name,"validation")[0]["macro_AP"]
    permutation=[]
    for name in feature_names[:64]:
        permuted=sample.copy();permuted[name]=rng.permutation(permuted[name].to_numpy())
        probability=predict_bundle(selected["bundle"],permuted,permuted.episode_id.to_numpy())
        score=metrics(sample_y,probability,thresholds,targets,best_name,"validation")[0]["macro_AP"]
        permutation.append({"attribute":name,"macro_AP_drop":base_ap-score,"rows":len(sample),"partition":"validation_explanation_only"})
    pd.DataFrame(permutation).to_csv(output/"permutation_importance.csv",index=False)
    te[["engine_id","episode_id","step","overall_health","subsystem_health_json","root_cause_confirmed","rul_status"]].to_csv(output/"health_assessment.csv",index=False)
    result_manifest={"source_csv":str(csv_path),"source_sha256":digest(csv_path),"source_schema_sha256":digest(Path(csv_path).with_suffix('.schema.json')),
                     "script_sha256":digest(__file__),"seed":seed,"epochs_budget":epochs,"selected_model":best_name,"selected_DL":best_dl,
                     "selection_policy":"highest validation macro AP; tree subset smallest within absolute0.015 of best, then full-fit finalist",
                     "train_rows":len(tr),"validation_tuning_rows":len(va),"calibration_rows":len(ca),"test_rows":len(te),
                     "calibration_truck":calibration_truck,"split_membership":manifest["split_membership"],
                     "tree_candidate_fit_rows":len(fit_indices),"elapsed_seconds":time.time()-started,
                     "versions":{"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"torch":dl.torch.__version__},
                     "source_dataset_limitations":"Synthetic, independent30minutejourneys, only3testtrucks, weakwearlabels may be undetectable, matrix alignment partial",
                     "input_exclusion_policy":exclusion_policy,
                     "test_evidence_status":"Previously used held-truck benchmark, not a newly untouched test" if exclude_gnss_brakes else "Initial held-truck benchmark",
                     "threshold_policy":"exact_validation_max_F1_with_1pct_per_label_normal_guard" if exclude_gnss_brakes else "exact_validation_max_F1",
                     "health":"Sensor-based unvalidated heuristic, not a DL health target",
                     "RUL":"unavailable_not_validated: no independent run-to-failure or censored histories",
                     "SHAP":"not_run: validation permutation and per-label feature ranks used instead",
                     "artifacts":{str(f.relative_to(output)):digest(f) for f in output.rglob('*') if f.is_file()}}
    (output/"manifest.json").write_text(json.dumps(result_manifest,indent=2))
    feature_contract={"source_input_columns":names,"selected_original_attributes":features,"model_input_columns":selected["bundle"]["input_columns"],
                      "input_exclusion_policy":exclusion_policy,
                      "target_columns":targets,"subsets":selection["subsets"],"preprocessing":"train-only median imputation and scaling where applicable",
                      "calibration":"sigmoid using one held validation truck; thresholds tuned on other two validation trucks",
                      "sensor_count_warning":"Feature count is not sensor count; context/reference dependencies still required",
                      "selected_model_bundle":"selected_model.joblib","selected_DL_model":calibrated[best_dl]["path"]}
    (output/"feature_schema.json").write_text(json.dumps(feature_contract,indent=2))
    report=["# Enhanced synthetic multi-label fault training results","",f"Selected overall model: **{best_name}**; best DL model: **{best_dl}**.",
            "", "Model selection used validation macro average precision. Test results were computed after freezing selection/calibration/thresholds.",
            "",markdown_table(pd.DataFrame(comparison)),"","## Limits",
            "Synthetic fault detection only; predictions do not confirm real root causes. Health is a sensor heuristic. RUL unavailable.",
            "Weak effects, partial matrix signatures, only three test trucks, route/time proxies and sensor masks limit generalization.",
            "Calibration uses one separate validation truck (not field calibration). AE/PCA dimensions do not imply fewer physical sensors.",
            "Missing-channel stress is partial masking of selected derived inputs, not a complete sensor-failure replay. SHAP was not run.",
             "", "## Reproduce", "```bash",f"OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_fault_models.py --output {output} --epochs {epochs}" + (" --exclude-gnss-brakes" if exclude_gnss_brakes else ""),"```"]
    (output/"REPORT.md").write_text('\n'.join(report)+'\n')
    return result_manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,default=ROOT/"datasets/dump_truck_enhanced_all_conditions.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"analysis/results_enhanced_multilabel")
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--seed",type=int,default=793)
    parser.add_argument("--exclude-gnss-brakes",action="store_true",help="Remove GNSS/GPS/brake inputs and derivatives before selection, PCA and training")
    args=parser.parse_args()
    print(json.dumps(run(args.csv,args.output,args.epochs,args.seed,args.exclude_gnss_brakes),indent=2))
