# Model comparisons

Data and saved results for Fig. 2b and Fig. 2c.

- `emgnn/`: paired predictions and statistical results for Fig. 2b.
- `sefgnn/`: saved results for six network tasks, with five runs per task.

The EMGNN paired analysis uses the mean prediction across five runs for each shared test record. AUPRC differences were assessed with 10,000 paired stratified bootstrap samples.

The GC-BRB record-level mean AUPRC is 0.8521. The primary identity-group mean AUPRC is 0.8533. These values use different evaluation protocols.

The original EMGNN bootstrap script is not included.
