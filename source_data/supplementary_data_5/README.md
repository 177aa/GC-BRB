# Threshold and support-rate summaries

Data for Supplementary Tables S82-S84. The inputs/ folder contains saved model results and predictions.

rebuild_threshold_summary.py summarizes validation thresholds, fixed-threshold test results and support-rate sensitivity using the recorded predictions and thresholds.

Validation precision intervals use the recorded thresholds and do not adjust for threshold selection.

Requirements: Python, pandas, NumPy and SciPy.
