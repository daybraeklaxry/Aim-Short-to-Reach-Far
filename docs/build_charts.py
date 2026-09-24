"""Build the website's two mean-success charts from the released CSV files."""
from pathlib import Path
from statistics import mean
from html import escape
import csv
import json
from math import log

DOCS = Path(__file__).resolve().parent
DATA = DOCS.parent / "data"
OUTPUT = DOCS / "assets" / "charts"
BLUE, GRAY, INK, GRID = "#0F4D92", "#767676", "#272727", "#deded8"
TASKS = {"cube", "pusht", "reacher", "tworoom"}
ARMS = {"cem_final": "Aiming at the goal", "cem_observed": "Aiming short"}
rows = list(csv.DictReader((DATA / "interventions.csv").open(encoding="utf-8")))
main = list(csv.DictReader((DATA / "main_results.csv").open(encoding="utf-8")))


def aggregate(kind, column, settings):
    result = {"x": settings, "series": {}}
    for arm, label in ARMS.items():
        points = []
        for setting in settings:
            selected = [r for r in rows if r["kind"] == kind and r["start"] == "Standard"
                        and r["config_arm"] == arm and r[column] == str(setting)]
            assert len(selected) == 4 and {r["task"] for r in selected} == TASKS
            assert all(r["complete"] == "true" and r["completed"] == r["expected"] == "128" for r in selected)
            values = {r["task"]: float(r["success_percent"]) for r in selected}
            points.append({"x": setting, "mean": mean(values.values()), "tasks": values})
        result["series"][arm] = {"label": label, "points": points}
    return result


distance = aggregate("horizon", "config_H", [25, 50, 100])
search = aggregate("budget", "config_iterations", [1, 2, 5, 10, 30])
gaps = [b["mean"] - a["mean"] for a, b in zip(distance["series"]["cem_final"]["points"], distance["series"]["cem_observed"]["points"])]
assert [round(gaps[i], 1) for i in [0, 2]] == [33.4, 47.9]
for arm, paper_name, rounded in [("cem_final", "CEM (final goal)", 9.2), ("cem_observed", "AP-CEM", 59.8)]:
    expected = mean(float(r["success_percent"]) for r in main if r["start"] == "Standard" and r["arm"] == paper_name)
    actual = search["series"][arm]["points"][-1]["mean"]
    assert actual == expected and round(actual, 1) == rounded
assert all(search["series"]["cem_observed"]["points"][1]["tasks"][task] >
           search["series"]["cem_final"]["points"][-1]["tasks"][task] for task in TASKS)


def svg_chart(data, xlabel, mobile=False, logarithmic=False):
    width, height = (350, 354) if mobile else (700, 380)
    left, right, top, bottom = (43, 307, 51, 292) if mobile else (56, 644, 50, 316)
    font = 17 if mobile else 18
    xs = data["x"]
    # Only the search-budget axis is logarithmic; no measured value changes.
    transform = log if logarithmic else lambda value: value
    x = lambda value: left + (transform(value) - transform(xs[0])) / (transform(xs[-1]) - transform(xs[0])) * (right - left)
    y = lambda value: bottom - value / 100 * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">Average success by {escape(xlabel.lower())}</title>',
             '<desc id="desc">Each point averages the four tasks with unchanged recorded starts. Aiming short is blue; aiming at the goal is gray.</desc>',
             f'<g font-family="Inter, Arial, sans-serif" font-size="{font}" fill="{INK}">']

    def text(xp, yp, content, color=INK, anchor="start", weight=400, extra=""):
        parts.append(f'<text x="{xp:g}" y="{yp:g}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{escape(content)}</text>')

    text(left, 25, "Average success (%)")
    for v in [0, 25, 50, 75, 100]:
        parts.append(f'<path d="M {left} {y(v):g} H {right}" fill="none" stroke="{GRID}" stroke-width="1"/>')
        text(left - 10, y(v) + 5, str(v), anchor="end")
    for v in xs:
        parts.append(f'<path d="M {x(v):g} {bottom} v 5" stroke="{GRAY}"/>')
        text(x(v), bottom + 26, str(v), anchor="middle")
    text((left + right) / 2, height - 7, xlabel, anchor="middle")
    for arm, color in [("cem_final", GRAY), ("cem_observed", BLUE)]:
        series = data["series"][arm]
        coords = [(x(p["x"]), y(p["mean"])) for p in series["points"]]
        dash = 'stroke-dasharray="7 5"' if arm == "cem_final" else ""
        parts.append(f'<polyline points="{" ".join(f"{xp:g},{yp:g}" for xp,yp in coords)}" fill="none" stroke="{color}" stroke-width="{2.5 if mobile else 3}" {dash} stroke-linejoin="round"/>')
        for xp, yp in coords:
            parts.append(f'<circle cx="{xp:g}" cy="{yp:g}" r="{3.7 if mobile else 5}" fill="{color}"/>')
        xp, yp = coords[-1]
        if not (logarithmic and arm == "cem_final"):
            text(xp, yp - 18,
                 series["label"], "#62625e" if arm == "cem_final" else color,
                 anchor="end", weight=500)
    if logarithmic:
        reference = data["series"]["cem_final"]["points"][-1]["mean"]
        parts.append(f'<path class="reference-line" d="M {left} {y(reference):g} H {right}" fill="none" stroke="{GRAY}" stroke-width="1" stroke-dasharray="4 5"/>')
        text(right, y(reference)-17, "30 iterations, aiming at the goal", "#62625e", anchor="end",
             extra=f'font-size="{17 if mobile else 16}"')
    parts.append('</g></svg>')
    return "\n".join(parts) + "\n"


OUTPUT.mkdir(exist_ok=True)
for name, data, xlabel in [("distance", distance, "Goal distance (actions)"), ("search", search, "Search iterations (log scale)")]:
    for mobile in [False, True]:
        (OUTPUT / f'{name}{"-mobile" if mobile else ""}.svg').write_text(svg_chart(data, xlabel, mobile, logarithmic=name=="search"), encoding="utf-8")
verification = {"source": "data/interventions.csv", "aggregation": "Equal-weight mean over the four tasks; Standard starts; unrounded success_percent.",
                "distance": distance, "search": search, "checks": {"distance_gaps_25_100": [gaps[0], gaps[2]],
                "search_at_30": [search["series"][a]["points"][-1]["mean"] for a in ARMS], "two_beats_thirty_on_each_task": True}}
(OUTPUT / "data.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"assertions":"passed", "search_x_scale":"log", **verification["checks"]}))
