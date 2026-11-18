function cfg = final_emgnn_stats_config(project_root)
%FINAL_EMGNN_STATS_CONFIG Configuration for final statistical comparison.

cfg = struct();

cfg.project_root = project_root;
cfg.gc_result_dir = fullfile( ...
    project_root,'results','final','official_exact');

% Leave empty for automatic detection. If automatic detection fails,
% MATLAB opens a folder-selection window once.
cfg.emgnn_table1_dir = '';

cfg.targets = {
    'CPDB'
    'MULTINET'
    'PCNET'
    'STRINGdb'
    'IREF'
    'IREF_2015'
};

% EMGNN used STRING as the directory prefix for STRINGdb.
cfg.emgnn_prefixes = {
    'CPDB'
    'MULTINET'
    'PCNET'
    'STRING'
    'IREF'
    'IREF_2015'
};

cfg.num_runs = 5;
cfg.emgnn_seeds = 72:76;

cfg.bootstrap_resamples = 10000;
cfg.bootstrap_seed = 20260806;
cfg.confidence_level = 0.95;
cfg.significance_level = 0.05;

% Strict checks required for a valid paired comparison.
cfg.require_exact_gene_set = true;
cfg.require_exact_labels = true;

cfg.output_dir = fullfile( ...
    project_root,'results','paper_statistics_emgnn');
end
