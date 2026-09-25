"""Recreate the released tables from compact success records, without simulators."""
from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from .statistics import ARMS, CONDITIONS, N, TASKS, estimate_tables


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def queries(protocol):
    result = protocol["studies"]
    for task in TASKS:
        rows = result[task]["confirm"]
        if len(rows) != N or len({q["record_id"] for q in rows}) != N:
            raise ValueError(f"{task}: expected {N} distinct frozen queries")
        if len({q["episode"] for q in rows}) != N:
            raise ValueError(f"{task}: the bootstrap requires distinct episode clusters")
    return result


def main_tensor(records, protocol):
    studies = queries(protocol)
    if tuple(protocol["arms"]) != ARMS or tuple(protocol["conditions"]) != CONDITIONS:
        raise ValueError("Main-study arm/condition order differs from the frozen analysis")
    indexed = {}
    for row in records:
        key = row["task"], row["record_id"]
        if key in indexed:
            raise ValueError(f"Duplicate query: {key}")
        indexed[key] = row["success"]
    expected = {(t, q["record_id"]) for t in TASKS for q in studies[t]["confirm"]}
    if set(indexed) != expected:
        raise ValueError(f"Expected all {len(expected)} frozen main-study queries")
    y = np.asarray([[indexed[t, q["record_id"]] for q in studies[t]["confirm"]] for t in TASKS])
    if y.shape != (4, N, 3, 5) or not np.isin(y, (0, 1)).all():
        raise ValueError("Success input must be a binary [4,128,3,5] tensor")
    return y.astype(np.float64)


def main_tables(y):
    summary = estimate_tables(y)
    rates = []
    for cell in summary["cells"]:
        for arm in ARMS:
            lo, hi = cell["success_ci95_percent"][arm]
            rates.append(dict(task=cell["task"], condition=cell["condition"], arm=arm,
                n_episode_clusters=N, success_percent=cell["success_percent"][arm],
                ci95_lower_percent=lo, ci95_upper_percent=hi,
                interval="empirical episode-cluster percentile bootstrap"))
    task_effects = []
    for row in summary["paired_contrasts"]:
        lo, hi = row["ci95_pp"]
        task_effects.append(dict(task=row["task"], condition=row["condition"], left=row["left"],
            right=row["right"], n_episode_clusters=N, difference_pp=row["difference_pp"],
            ci95_lower_pp=lo, ci95_upper_pp=hi))
    macro_rates = []
    for row in summary["equal_task_macro"]:
        for arm in ARMS:
            lo, hi = row["success_ci95_percent"][arm]
            macro_rates.append(dict(condition=row["condition"], arm=arm, task_weight=.25,
                n_episode_clusters_per_task=N, success_percent=row["success_percent"][arm],
                ci95_lower_percent=lo, ci95_upper_percent=hi))
    macro_effects = []
    for row in summary["macro_contrasts"]:
        lo, hi = row["ci95_pp"]
        macro_effects.append(dict(condition=row["condition"], left=row["left"], right=row["right"],
            prespecified_role=row["prespecified_role"], difference_pp=row["difference_pp"],
            ci95_lower_pp=lo, ci95_upper_pp=hi))
    return {"success_rates.csv": rates, "task_effects.csv": task_effects,
            "macro_rates.csv": macro_rates, "macro_effects.csv": macro_effects}, summary


def local_tables(paired, protocol):
    studies = queries(protocol)
    by_key = {}
    for row in paired:
        key = row["task"], row["record_id"]
        if key in by_key:
            raise ValueError(f"Duplicate paired query: {key}")
        by_key[key] = row
    expected = {(t, q["record_id"]) for t in TASKS for q in studies[t]["confirm"]}
    if set(by_key) != expected:
        raise ValueError(f"Expected all {len(expected)} frozen local-target queries")
    rates, details, effects = [], [], []
    estimates = {}
    for task in TASKS:
        task_rows = [by_key[task, q["record_id"]] for q in studies[task]["confirm"]]
        for condition in CONDITIONS:
            for arm in ("anchor", "transport"):
                values = [int(row[f"{condition}_{arm}"]) for row in task_rows]
                if any(value not in (0, 1) for value in values):
                    raise ValueError("Success values must be binary")
                estimate = Fraction(sum(values), N)
                estimates[task, condition, arm] = estimate
                details.append(dict(task=task, condition=condition, arm=arm, queries=N,
                                    successes=sum(values), success_percent=float(100 * estimate)))
            difference = estimates[task, condition, "anchor"] - estimates[task, condition, "transport"]
            effects.append(dict(task=task, condition=condition, anchor_minus_transport_pp=float(100 * difference)))
        for arm in ("anchor", "transport"):
            estimates[task, "perturbed", arm] = (estimates[task, "prefix_a", arm]
                                                   + estimates[task, "prefix_b", arm]) / 2
        for setting, condition in (("standard", "clean"), ("perturbed", "perturbed")):
            a, b = (estimates[task, condition, arm] for arm in ("anchor", "transport"))
            rates.append(dict(task=task, start=setting, anchor_percent=float(100*a),
                transport_percent=float(100*b), anchor_minus_transport_pp=float(100*(a-b))))
    for setting, condition in (("standard", "clean"), ("perturbed", "perturbed")):
        a, b = (sum(estimates[t, condition, arm] for t in TASKS) / 4 for arm in ("anchor", "transport"))
        rates.append(dict(task="macro", start=setting, anchor_percent=float(100*a),
            transport_percent=float(100*b), anchor_minus_transport_pp=float(100*(a-b))))
    summary = dict(status="complete", assigned_outcomes=3072, queries_per_task=N, tasks=TASKS,
        arms=("anchor", "transport"), primary_macro_effects=rates[-2:],
        aggregation="Average prefixes within query, queries within task, then tasks equally.",
        rounding="Success effects are calculated from exact fractions before display rounding.",
        scope="New paired target comparison; no pooling with the earlier main cohort.")
    return {"main_rates.csv": rates, "separate_start_rates.csv": details,
            "separate_start_effects.csv": effects}, summary


def records_from_outcomes(directory, protocol, study):
    """Read complete fresh runs; shard files can be collected in one directory."""
    studies = queries(protocol)
    arms = ARMS if study == "main" else ("anchor", "transport")
    expected = {(t, q["record_id"], c, a) for t in TASKS for q in studies[t]["confirm"]
                for c in CONDITIONS for a in arms}
    indexed = {}
    for path in sorted(Path(directory).glob("outcomes_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["task"], row["identity"]["record_id"], row["condition"], row["arm"]
                if key not in expected or key in indexed:
                    raise ValueError(f"Unexpected or duplicate outcome: {key}")
                if not row["completed"] or row.get("profile_censored", False) or type(row["success"]) is not bool:
                    raise ValueError(f"Unfinished or invalid outcome: {key}")
                indexed[key] = row["success"]
    if set(indexed) != expected:
        raise ValueError(f"Incomplete frozen cohort: {len(indexed)}/{len(expected)} outcomes")
    records = []
    for task in TASKS:
        for query in studies[task]["confirm"]:
            qid = query["record_id"]
            row = dict(task=task, record_id=qid)
            if study == "main":
                row["success"] = [[int(indexed[task, qid, c, a]) for a in arms] for c in CONDITIONS]
            else:
                row["episode"] = query["episode"]
                row.update({f"{c}_{a}": int(indexed[task, qid, c, a]) for c in CONDITIONS for a in arms})
            records.append(row)
    return records


def check_tables(tables, reference):
    """Compare every released field; fail on a changed aggregation or interval."""
    checked = 0
    for name, actual in tables.items():
        expected = read_csv(Path(reference) / name)
        if len(actual) != len(expected):
            raise ValueError(f"{name}: changed row count")
        for index, (a, b) in enumerate(zip(actual, expected)):
            if set(a) != set(b):
                raise ValueError(f"{name}: changed columns")
            for key, value in a.items():
                same = np.isclose(float(value), float(b[key]), rtol=0, atol=1e-10) if isinstance(value, (int, float)) else str(value) == b[key]
                if not same:
                    raise ValueError(f"{name}, row {index}, {key}: {value!r} != {b[key]!r}")
            checked += 1
    return checked


def run(args):
    study = args.study.replace("-", "_")
    protocol = read_json(args.root / "protocols" / f"{study}.json")
    records = records_from_outcomes(args.outcomes, protocol, study)
    if study == "main":
        tables, summary = main_tables(main_tensor(records, protocol))
    else:
        tables, summary = local_tables(records, protocol)
    for name, rows in tables.items():
        write_csv(args.output / name, rows)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(study=study, tables=len(tables), rows=sum(map(len, tables.values())),
        output=str(args.output), source="evaluation outcomes")))
