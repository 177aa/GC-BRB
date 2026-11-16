function cfg = gg_config(stage)
%GG_CONFIG The source path below is the only machine-specific setting.
if nargin < 1, stage = 'preflight'; end
root = fileparts(mfilename('fullpath'));
cfg = struct();
cfg.protocol_version = 'GG_OOF_1_0';
cfg.stage = char(stage);
cfg.project_root = root;
cfg.source_project_root = fullfile(root,'model_input');
sibling = fullfile(fileparts(root),'GC_BRB_3LAYER_FINAL');
if ~isfolder(cfg.source_project_root) && isfolder(sibling)
    cfg.source_project_root = sibling;
end
cfg.prepared_path = fullfile(cfg.source_project_root,'cache','prepared', ...
    'multilayer_aligned_gcbrb.mat');
cfg.result_root = fullfile(root,'results',cfg.protocol_version,cfg.stage);
cfg.all_datasets = {'CPDB';'MULTINET';'PCNET';'STRINGdb';'IREF';'IREF_2015'};
cfg.target_datasets = cfg.all_datasets;
cfg.num_runs = 5;
cfg.seed_base = 20260712;
cfg.identity_rule = 'connected_nonempty_ensembl_OR_upper_symbol';
cfg.validation_fraction = 0.10;
cfg.outer_folds = 3;
cfg.crossfit_folds = 3;
cfg.omics_tuning_folds = 3;
cfg.omics_standardize = true;
cfg.omics_lambda_grid = [1e-6 1e-5 1e-4 1e-3 1e-2];
cfg.lr_lambda_grid = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];
cfg.fusion_alpha_grid = 0:0.05:1;
cfg.threshold_metric = 'MCC';
cfg.high_precision_target = 0.95;
cfg.high_precision_min_calls = 5;
cfg.rwr_restart = 0.20;
cfg.rwr_max_iterations = 150;
cfg.rwr_tolerance = 1e-10;
cfg.neighbor_smoothing = 1e-12;
cfg.eps = 1e-12;
cfg.max_generations = 190;
cfg.early_stop_patience = 40;
cfg.cmaes_restarts = 2;
cfg.evaluate_parameter_consensus = true;
cfg.sigma0 = 0.11;
cfg.population_size = [];
cfg.loss = 'balanced_bce_rank';
cfg.ranking_loss_weight = 0.20;
cfg.ranking_pairs_per_positive = 8;
cfg.ranking_temperature = 0.10;
cfg.structure_version = 'V3LAYER_1_0';
cfg.verbose = true;
cfg.crossfit_verbose = false;
cfg.resume = true;
cfg.export_baseline_features = true;
switch cfg.stage
    case 'preflight'
        % All 30 splits are audited; no feature construction or training.
    case 'smoke'
        cfg.target_datasets = {'CPDB'};
        cfg.num_runs = 1;
        cfg.max_generations = 3;
        cfg.early_stop_patience = 2;
        cfg.cmaes_restarts = 1;
        cfg.evaluate_parameter_consensus = false;
        cfg.population_size = 6;
        cfg.rwr_max_iterations = 20;
    case 'pilot'
        cfg.target_datasets = {'CPDB'};
        cfg.num_runs = 1;
    case 'formal'
        % Six fixed test tasks, five new grouped train/validation splits.
    otherwise
        error('GG:Stage','Use preflight, smoke, pilot or formal.');
end
if any(strcmp(cfg.stage,{'pilot','formal'}))
    cfg.profile='formal';
else
    cfg.profile=cfg.stage;
end
% The formal CPDB pilot is run 1 of the same 30-run study, so it is reusable.
cfg.result_root=fullfile(root,'results',cfg.protocol_version,cfg.profile);
end
