%% Final three-layer GC-BRB versus EMGNN statistical comparison
% This entry does NOT retrain the model and does NOT overwrite main results.
%
% Required:
%   1. Final GC-BRB predictions already exist under:
%      results/final/official_exact/<TARGET>/run_XX/test_predictions.csv
%   2. EMGNN replicated predictions exist under:
%      outputs/table1/<TARGET>_seed72...76/test_predictions.csv
%
% Run:
%   clear; clc; main_compare_emgnn

clc
clear
close all

project_root = fileparts(mfilename('fullpath'));
addpath(project_root);
addpath(fullfile(project_root,'core'));

cfg = final_emgnn_stats_config(project_root);

fprintf('\n============================================================\n');
fprintf('FINAL THREE-LAYER GC-BRB vs EMGNN\n');
fprintf('AUROC: paired DeLong test\n');
fprintf('AUPRC: paired stratified bootstrap (%d resamples)\n', ...
    cfg.bootstrap_resamples);
fprintf('Multiplicity: Holm correction across six PPI networks\n');
fprintf('============================================================\n');

run_final_emgnn_statistics(cfg);
