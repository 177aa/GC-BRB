Threshold and support-rate summaries
This folder contains the data for Supplementary Tables S82-S84.
The inputs folder contains the saved model results and predictions. The script rebuild\_threshold\_summary.py uses these files to generate the validation threshold, fixed-threshold test and support-rate sensitivity summaries.

The script does not train models or select new thresholds. The validation precision intervals do not account for threshold selection.
Requirements: Python, pandas, NumPy and SciPy.
