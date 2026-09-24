# Supporting result reproduction

These saved statistics regenerate 11 rows from earlier supporting studies, rather than the current manuscript tables. Current results are in the repository's `data/` directory. Each study retains its own cohort and action protocol; these results are not pooled with the main or additional local-target study.

From the repository root, run:

```sh
python code/ap/examples/reproduce_supporting_tables.py --output outputs/supporting
```

`confirmation_diagnostics.json` contains the endpoint-error and local-regret estimates from the earlier diagnostic study. The script rounds the original estimates to three decimals. `native_controller_comparison.json` supplies the success and prediction-work values from the earlier controller comparison. `prediction_budget_curve.csv` supplies the earlier search-budget success values under the assigned CEM budgets. Success rates are displayed as percentages with one decimal.

Original statistics retain their source field names. The `released` key in `native_controller_comparison.json` identifies the native LeWM controller. Prediction work counts five-action blocks and is not a latency measurement. These files contain statistics rather than benchmark observations or model parameters.
