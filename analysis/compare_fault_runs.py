"""Read-only comparison of saved fault runs; no training or model reselection.

``compare(old_dir, new_dir, output_prefix=None)`` returns a JSON-compatible
report and atomically replaces its Markdown/JSON reports in ``new_dir`` only.
The default prefix is ``INPUT_EXCLUSION_COMPARISON``. Missing scores remain
null, and deltas are withheld when the evaluation populations do not match.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile


METRICS = ("macro_AP", "micro_AP", "macro_F1", "macro_recall", "micro_F1",
           "hamming_loss", "exact_match", "normal_false_alarm_rate", "macro_Brier")
FAULT_METRICS = ("AP", "precision", "recall", "F1", "support", "threshold",
                 "TP", "FP", "FN", "TN", "Brier")
INPUT_BRANCHES = ("source_input_columns", "model_input_columns",
                  "selected_original_attributes", "subsets")
EXCLUDED = {"gnss", "brake", "gps", "braking"}
CAVEATS = [
    "Initial raw and sigmoid-calibrated scores are same held-trucks benchmark scores, "
    "not a newly untouched test: the prior run already tested these trucks.",
    "Candidates are selected by manifest.selected_model (validation policy), never by test metrics.",
    "A reused independent seed is a benchmark, not fresh unseen evidence for this comparison; "
    "synthetic engine variants and route families are not field validation.",
    "brake_wear remains a target, but without direct brake readings its classifier is unsupported; "
    "proxy scores do not establish brake condition or confirmed root cause.",
    "Perception LiDAR/radar inputs remain eligible; their presence in the new schema is reported below.",
    "RUL is not validated: no independent run-to-failure or censored histories establish remaining life.",
    "Guarded threshold results are separate post-hoc, second-use test evidence for human interpretation; "
    "a per-label validation alarm cap is not an overall or field false-alarm guarantee.",
    "Missing/null/non-finite metrics are reported as missing, never as zero. "
    "Feature counts are input dimensions, not physical sensor counts.",
]


def _json(path, optional=False):
    if optional and not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path, optional=False):
    if optional and not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _names(value, branch):
    """Inspect every subset recursively, including features in nested subsets."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _names(child, f"{branch}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _names(child, f"{branch}[{index}]")
    elif isinstance(value, str):
        yield branch, value
    else:
        raise ValueError(f"Invalid input-name branch {branch}: expected input names")


def _verify_exclusion(schema):
    violations = []
    for branch in INPUT_BRANCHES:
        if branch not in schema:
            raise ValueError(f"New schema missing required input branch: {branch}")
        if not isinstance(schema[branch], dict if branch == "subsets" else list):
            raise ValueError(f"Invalid input-name branch {branch}")
        for location, name in _names(schema[branch], branch):
            tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
            if tokens & EXCLUDED:
                violations.append(f"{location}: {name}")
    if violations:
        raise ValueError("Excluded input names in new schema: " + "; ".join(violations))
    return {"passed": True, "checked_branches": list(INPUT_BRANCHES),
            "excluded_name_tokens": sorted(EXCLUDED), "targets_checked_as_inputs": False}


def _check(old, new):
    if old is None or new is None or old == [] or new == [] or old == {} or new == {}:
        status = "missing"
    else:
        status = "same" if old == new else "different"
    return {"old": old, "new": new, "status": status}


def _splits(manifest):
    membership = manifest.get("split_membership")
    if not isinstance(membership, dict):
        return None
    return {key: sorted(value) for key, value in membership.items()}


def _comparability(old, new):
    checks = {
        "source_csv_sha256": _check(old["manifest"].get("source_sha256"), new["manifest"].get("source_sha256")),
        "split_membership": _check(_splits(old["manifest"]), _splits(new["manifest"])),
        "calibration_truck": _check(old["manifest"].get("calibration_truck"), new["manifest"].get("calibration_truck")),
        "target_columns": _check(old["schema"].get("target_columns"), new["schema"].get("target_columns")),
    }
    return {"comparable": all(check["status"] == "same" for check in checks.values()),
            "checks": checks, "evidence_label": "same held-trucks benchmark; prior run already tested"}


def _row(rows, candidate, split="test", fault=None):
    matches = [row for row in rows if row.get("candidate") == candidate and row.get("split") == split
               and (fault is None or row.get("fault") == fault)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous saved metrics: {candidate}, {split}, {fault}")
    return matches[0] if matches else None


def _scores(row, keys=METRICS):
    return {key: _number(row.get(key)) if row else None for key in keys}


def _delta(old, new, comparable):
    return {key: new[key] - old[key] if comparable and old[key] is not None and new[key] is not None
            else None for key in old}


def _pair(old, new, comparable, keys=METRICS):
    before, after = _scores(old, keys), _scores(new, keys)
    return {"old": before, "new": after, "delta_new_minus_old": _delta(before, after, comparable),
            "missing_metrics": {"old": [key for key in keys if before[key] is None],
                                "new": [key for key in keys if after[key] is None]},
            "old_row_present": old is not None, "new_row_present": new is not None}


def _load(directory):
    manifest = _json(directory / "manifest.json")
    schema = _json(directory / "feature_schema.json")
    selected = manifest.get("selected_model")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"Missing manifest.selected_model in {directory}")
    return {"directory": directory, "manifest": manifest, "schema": schema,
            "selected_model": selected, "comparison": _csv(directory / "model_comparison.csv"),
            "per_fault": _csv(directory / "per_fault_metrics.csv", optional=True),
            "ranking": _json(directory / "feature_selection/selected_preliminary_ranks.json", optional=True),
            "independent": _json(directory / "independent_seed_validation/RESULTS.json", optional=True),
            "guarded": _csv(directory / "threshold_refinement_comparison.csv", optional=True)}


def _features(run):
    schema, ranking = run["schema"], run["ranking"]
    result = {key: schema.get(key) for key in INPUT_BRANCHES if key != "subsets"}
    result["counts"] = {key: len(schema[key]) if isinstance(schema.get(key), list) else None
                        for key in INPUT_BRANCHES if key != "subsets"}
    for key in ("eligible_names", "pruned_names"):
        names = ranking.get(key) if ranking else None
        result["counts"][key] = len(names) if isinstance(names, list) else None
    result["pruned_count_semantics"] = "pruned_names counts retained inputs after correlation pruning, not removed inputs"
    result["ranking_source"] = "feature_selection/selected_preliminary_ranks.json"
    result["ranking_missing"] = ranking is None
    result["perception_inputs"] = [name for name in schema.get("source_input_columns", [])
                                   if {"lidar", "radar"} & set(re.findall(r"[a-z0-9]+", name.lower()))]
    return result


def _phase_candidate(run, phase):
    candidate = run["selected_model"] + ("_raw" if phase == "raw" else "_sigmoid_calibrated")
    # Older artifacts may call the raw selected model by its unsuffixed name.
    if phase == "raw" and _row(run["comparison"], candidate) is None:
        candidate = run["selected_model"]
    return candidate


def _independent(old, new):
    before, after = old["independent"], new["independent"]
    checks = {"seed": _check(before.get("seed") if before else None, after.get("seed") if after else None),
              "target_columns": _check(old["schema"].get("target_columns"), new["schema"].get("target_columns"))}
    for key in ("independent_engine_variant_seeds", "rows"):
        checks[key] = _check(before.get(key) if before else None, after.get(key) if after else None)
    def batch_hash(record):
        return (record or {}).get("receipt", {}).get("sha256", {}).get("independent_fault_batch.csv")
    checks["batch_csv_sha256"] = _check(batch_hash(before), batch_hash(after))
    comparable = all(check["status"] == "same" for check in checks.values())
    same_seed = checks["seed"]["status"] == "same"
    result = {"available": {"old": before is not None, "new": after is not None},
              "comparable": comparable, "checks": checks, "seed_reused": same_seed,
              "evidence_label": "reused-seed benchmark, not fresh unseen evidence" if same_seed
              else "different or missing seeds: no matched independent-seed benchmark established",
              "scores": _pair(before.get("score") if before else None,
                              after.get("score") if after else None, comparable),
              "frozen_policy_receipts": {side: {key: record.get(key) if record else None for key in
                  ("weights_and_thresholds_frozen_before_generation", "training_or_tuning_performed")}
                  for side, record in (("old", before), ("new", after))}}
    targets = list(dict.fromkeys(old["schema"].get("target_columns", []) + new["schema"].get("target_columns", [])))
    fault_rows = [_csv(run["directory"] / "independent_seed_validation/per_fault_metrics.csv", optional=True)
                  for run in (old, new)]
    result["per_fault"] = []
    for fault in targets:
        rows = []
        for record, saved in zip((before, after), fault_rows):
            score = (record or {}).get("score", {})
            rows.append(_row(saved, score.get("candidate"), score.get("split"), fault))
        result["per_fault"].append({"fault": fault, **_pair(*rows, comparable, FAULT_METRICS)})
    return result


def _guarded(old, new, comparable):
    pairs = sorted({(row.get("candidate"), row.get("split")) for run in (old, new) for row in run["guarded"]})
    return {"evidence_label": "post-hoc second-use, separate from initial test; human interpretation only",
            "rows": [{"candidate": candidate, "split": split,
                      **_pair(_row(old["guarded"], candidate, split), _row(new["guarded"], candidate, split), comparable)}
                     for candidate, split in pairs]}


def _display(value):
    if value is None:
        return "missing"
    return f"{value:.6g}" if isinstance(value, (int, float)) else str(value).replace("|", "\\|").replace("\n", " ")


def _table(pair, keys=METRICS):
    lines = ["| Metric | Old | New | New − old |", "| --- | ---: | ---: | ---: |"]
    for key in keys:
        lines.append("| " + " | ".join(_display(value) for value in
                     (key, pair["old"][key], pair["new"][key], pair["delta_new_minus_old"][key])) + " |")
    return lines


def _markdown(report):
    lines = ["# Input exclusion comparison", "", "## Interpretation", ""]
    lines.extend("- " + caveat for caveat in report["caveats"])
    lines += ["", "## Comparability", "", f"Matched original benchmark: **{report['comparability']['comparable']}**."]
    for name, check in report["comparability"]["checks"].items():
        lines.append(f"- {name}: {check['status']}")
    lines += ["", "## Input features", "", "Exclusion verified across every new input branch and subset; targets are retained.", ""]
    for side in ("old", "new"):
        features = report["features"][side]
        lines.append(f"### {side.title()}: manifest-selected {_display(report['selected_models'][side])}")
        lines.extend(f"- {key}: {_display(value)}" for key, value in features["counts"].items())
        lines.append("- " + features["pruned_count_semantics"])
        lines.append("- LiDAR/radar eligible inputs: " + (", ".join(features["perception_inputs"]) or "none recorded"))
        for key in ("selected_original_attributes", "model_input_columns"):
            lines += ["", f"**{key}**", ""]
            lines.extend("- " + _display(name) for name in features[key] or [])
    lines += ["", "## Initial selected-model test scores", "", "Same held-trucks, prior-tested benchmark. Deltas withheld if not comparable."]
    for phase, pair in report["selected_test_scores"].items():
        lines += ["", f"### {phase}: {_display(pair['old_candidate'])} → {_display(pair['new_candidate'])}", ""] + _table(pair)
    lines += ["", "## Per-fault initial test scores", "", "Weak flags: AP, F1 or recall < 0.5 in either run; reporting only, never model selection.", "",
              "| Phase | Fault | Old AP | New AP | Old F1 | New F1 | Old recall | New recall | Weak / unsupported |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in report["per_fault"]:
        values = (row["phase"], row["fault"], row["old"]["AP"], row["new"]["AP"], row["old"]["F1"],
                  row["new"]["F1"], row["old"]["recall"], row["new"]["recall"], ", ".join(row["weak_reasons"]) or "none observed")
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    independent = report["independent_seed_comparison"]
    lines += ["", "## Optional independent-seed benchmark", "", independent["evidence_label"],
              f"Matched benchmark: **{independent['comparable']}**."]
    lines.extend(f"- {name}: {check['status']}" for name, check in independent["checks"].items())
    lines += [""] + _table(independent["scores"])
    lines += ["", "## Optional guarded thresholds (separate evidence)", "", report["guarded_threshold_comparison"]["evidence_label"]]
    guarded = report["guarded_threshold_comparison"]["rows"]
    if not guarded:
        lines += ["", "missing: threshold_refinement_comparison.csv results"]
    for row in guarded:
        lines += ["", f"### {row['candidate']} / {row['split']}", ""] + _table(row)
    lines += ["", "Full saved candidate and per-fault comparisons, including missing fields, are in the companion JSON."]
    return "\n".join(lines) + "\n"


def _atomic_write(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def compare(old_dir, new_dir, output_prefix=None):
    """Compare persisted artifacts and write only two reports inside the new run."""
    old_dir, new_dir = Path(old_dir).resolve(), Path(new_dir).resolve()
    if old_dir == new_dir:
        raise ValueError("Old and new run directories must differ")
    old, new = _load(old_dir), _load(new_dir)
    exclusion = _verify_exclusion(new["schema"])
    prefix = Path(output_prefix) if output_prefix is not None else Path("INPUT_EXCLUSION_COMPARISON")
    prefix = (prefix if prefix.is_absolute() else new_dir / prefix).resolve()
    if prefix.suffix.lower() in (".md", ".json"):
        prefix = prefix.with_suffix("")
    destinations = [Path(str(prefix) + extension) for extension in (".md", ".json")]
    for destination in destinations:
        resolved = destination.resolve()
        if not resolved.is_relative_to(new_dir) or resolved.is_relative_to(old_dir):
            raise ValueError("Comparison outputs must stay in new folder and cannot overwrite old artifacts")
        if not destination.parent.is_dir():
            raise ValueError("Comparison output parent folder must already exist")
        # Protect saved input artifacts even when a custom output prefix collides.
        if destination.exists() and destination.name != "INPUT_EXCLUSION_COMPARISON" + destination.suffix:
            existing = destination.read_text(encoding="utf-8")
            if (destination.suffix == ".json" and '"report_type": "input_exclusion_comparison"' not in existing
                    or destination.suffix == ".md" and not existing.startswith("# Input exclusion comparison\n")):
                raise ValueError(f"Refusing to overwrite existing input artifact: {destination}")
    comparability = _comparability(old, new)
    comparable = comparability["comparable"]
    report = {"report_type": "input_exclusion_comparison", "old_dir": str(old_dir), "new_dir": str(new_dir),
              "source_csv_paths": {side: run["manifest"].get("source_csv") for side, run in (("old", old), ("new", new))},
              "selected_models": {"old": old["selected_model"], "new": new["selected_model"]},
              "selection_policies": {side: run["manifest"].get("selection_policy") for side, run in (("old", old), ("new", new))},
              "exclusion_verification": exclusion, "comparability": comparability, "caveats": CAVEATS,
              "features": {"old": _features(old), "new": _features(new)}, "selected_test_scores": {}, "per_fault": []}
    targets = list(dict.fromkeys(old["schema"].get("target_columns", []) + new["schema"].get("target_columns", [])))
    for phase in ("raw", "sigmoid_calibrated"):
        candidates = [_phase_candidate(run, phase) for run in (old, new)]
        report["selected_test_scores"][phase] = {"old_candidate": candidates[0], "new_candidate": candidates[1],
            **_pair(_row(old["comparison"], candidates[0]), _row(new["comparison"], candidates[1]), comparable)}
        for fault in targets:
            pair = _pair(_row(old["per_fault"], candidates[0], fault=fault),
                         _row(new["per_fault"], candidates[1], fault=fault), comparable, FAULT_METRICS)
            reasons = [f"{side} {metric}<0.5" for side in ("old", "new") for metric in ("AP", "F1", "recall")
                       if pair[side][metric] is not None and pair[side][metric] < .5]
            if fault.rsplit(".", 1)[-1] == "brake_wear":
                reasons.append("unsupported without direct brake readings")
            report["per_fault"].append({"phase": phase, "fault": fault, "weak_reasons": reasons, **pair})
    report["weak_classes"] = sorted({row["fault"] for row in report["per_fault"] if row["weak_reasons"]})
    candidate_names = sorted({row["candidate"] for run in (old, new) for row in run["comparison"] if row.get("split") == "test"})
    report["all_candidate_test_scores"] = [{"candidate": name, **_pair(_row(old["comparison"], name),
        _row(new["comparison"], name), comparable)} for name in candidate_names]
    report["independent_seed_comparison"] = _independent(old, new)
    report["guarded_threshold_comparison"] = _guarded(old, new, comparable)
    report["output_files"] = {"markdown": str(destinations[0]), "json": str(destinations[1])}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown = _markdown(report)
    _atomic_write(destinations[1], encoded)
    _atomic_write(destinations[0], markdown)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.old_dir, args.new_dir, args.output_prefix)["output_files"], indent=2))
