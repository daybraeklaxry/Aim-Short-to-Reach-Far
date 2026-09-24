"""Recreate the diagnostic, native-controller, and search-budget table values."""
import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def rounded(value, digits=1):
    return format(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits),
                                              rounding=ROUND_HALF_UP), f".{digits}f")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(data, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((data / name).read_text(encoding="utf-8"))
    tasks = {"cube": "Cube", "pusht": "PushT", "reacher": "Reacher", "tworoom": "TwoRoom"}
    diagnostic = read("confirmation_diagnostics.json")
    metrics = {"predictor_endpoint_error": "learned_endpoint_mse",
               "displacement_endpoint_error": "delta_endpoint_mse",
               "predictor_local_regret": "learned_local_regret",
               "displacement_local_regret": "delta_local_regret",
               "direct_local_regret": "direct_local_regret"}
    rows = []
    for task, label in tasks.items():
        source = diagnostic["tasks"][task]
        row = {"task": label, "N": source["eligible_states"]}
        row.update({name: rounded(source["metrics"][key]["estimate"], 3)
                    for name, key in metrics.items()})
        rows.append(row)
    write_csv(output / "ranking_diagnostics.csv", rows)

    native = read("native_controller_comparison.json")
    rows = []
    for task, label in tasks.items():
        source = native["tasks"][task]
        n = Decimal(source["n"])
        totals = source["candidate_block_work"]["total"]
        rows.append({"task": label,
                     "native_success_percent": rounded(100 * Decimal(source["successes"]["released"]) / n),
                     "observed_success_percent": rounded(100 * Decimal(source["successes"]["local_cem"]) / n),
                     "native_mean_blocks": rounded(Decimal(totals["released"]) / n),
                     "observed_mean_blocks": rounded(Decimal(totals["local_cem"]) / n),
                     "relative_work_percent": rounded(100 * Decimal(totals["local_cem"]) / Decimal(totals["released"]))})
    write_csv(output / "native_controller.csv", rows)

    with (data / "prediction_budget_curve.csv").open(newline="", encoding="utf-8") as stream:
        budget = list(csv.DictReader(stream))
    lookup = {(r["task"], int(r["iterations"]), r["arm"]): r for r in budget}
    rows = []
    for iterations in (1, 5, 30):
        row = {"cem_iterations": iterations}
        for task in ("cube", "pusht"):
            for arm, label in (("far", "final_goal"), ("local", "observed")):
                record = lookup[task, iterations, arm]
                row[f"{task}_{label}_success_percent"] = rounded(
                    100 * Decimal(record["successes"]) / Decimal(record["episodes"]))
        rows.append(row)
    write_csv(output / "search_budget.csv", rows)
    print(json.dumps({"tables": 3, "rows": 11, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/supporting")
    parser.add_argument("--output", type=Path, default=Path("reproduced/supporting"))
    args = parser.parse_args()
    reproduce(args.data, args.output)
