function run_final_emgnn_statistics(cfg)
%RUN_FINAL_EMGNN_STATISTICS
% Identity fix v2: Ensembl-first alignment with missing-text handling.
%
% For each PPI network:
%   1. Align each GC-BRB run with its matching EMGNN seed by
%      Ensembl ID + Gene Symbol.
%   2. Verify gene sets and truth labels.
%   3. Average the five probabilities for each gene within each method.
%   4. Apply paired DeLong to the two averaged AUROC values.
%   5. Apply 10,000-resample paired stratified bootstrap to AUPRC delta.
%   6. Apply Holm correction across the six networks separately for
%      AUROC and AUPRC.
%
% Averaging the five run/seed predictions before sample-level inference
% avoids treating the same test gene as five independent observations.

if ~isfolder(cfg.gc_result_dir)
    error(['Final GC-BRB result directory was not found:\n  %s\n' ...
        'Run the final three-layer model first.'],cfg.gc_result_dir);
end

emgnn_dir = resolve_emgnn_table1_dir_local(cfg);

if ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

aligned_dir = fullfile(cfg.output_dir,'aligned_averaged_predictions');
if ~isfolder(aligned_dir)
    mkdir(aligned_dir);
end

fprintf('\nGC-BRB predictions:\n  %s\n',cfg.gc_result_dir);
fprintf('EMGNN predictions:\n  %s\n',emgnn_dir);
fprintf('Statistical output:\n  %s\n',cfg.output_dir);

audit_rows = struct([]);
run_metric_rows = struct([]);
result_rows = struct([]);

for target_i = 1:numel(cfg.targets)
    target = cfg.targets{target_i};
    emgnn_prefix = cfg.emgnn_prefixes{target_i};

    fprintf('\n============================================================\n');
    fprintf('Network %d/%d: %s\n', ...
        target_i,numel(cfg.targets),target);
    fprintf('============================================================\n');

    reference_keys = strings(0,1);
    reference_ensembl = strings(0,1);
    reference_symbols = strings(0,1);
    reference_labels = [];
    gc_score_matrix = [];
    emgnn_score_matrix = [];

    for run = 1:cfg.num_runs
        seed = cfg.emgnn_seeds(run);

        gc_file = fullfile(cfg.gc_result_dir,target, ...
            sprintf('run_%02d',run),'test_predictions.csv');

        emgnn_file = fullfile(emgnn_dir, ...
            sprintf('%s_seed%d',emgnn_prefix,seed), ...
            'test_predictions.csv');

        require_file_local(gc_file,'GC-BRB prediction');
        require_file_local(emgnn_file,'EMGNN prediction');

        G = read_gc_prediction_local(gc_file);
        E = read_emgnn_prediction_local(emgnn_file);

        [A,audit] = align_prediction_pair_local( ...
            G,E,target,run,seed,cfg);

        audit_rows = append_struct_local(audit_rows,audit);

        if run == 1
            reference_keys = A.Key;
            reference_ensembl = A.EnsemblID;
            reference_symbols = A.GeneSymbol;
            reference_labels = A.Label;
            gc_score_matrix = nan(height(A),cfg.num_runs);
            emgnn_score_matrix = nan(height(A),cfg.num_runs);
        else
            if ~isequal(reference_keys,A.Key)
                error(['The aligned test gene order/set differs across ' ...
                    'runs for %s.'],target);
            end
            if ~isequal(reference_labels,A.Label)
                error(['The truth labels differ across runs for %s.'], ...
                    target);
            end
        end

        gc_score_matrix(:,run) = A.GCBRBProbability;
        emgnn_score_matrix(:,run) = A.EMGNNProbability;

        gc_metrics = binary_metrics( ...
            A.Label,A.GCBRBProbability,0.5);
        emgnn_metrics = binary_metrics( ...
            A.Label,A.EMGNNProbability,0.5);

        rr = struct();
        rr.Network = string(target);
        rr.Run = run;
        rr.EMGNNSeed = seed;
        rr.NumTestGenes = height(A);
        rr.GCBRB_AUROC = gc_metrics.AUROC;
        rr.EMGNN_AUROC = emgnn_metrics.AUROC;
        rr.DeltaAUROC = gc_metrics.AUROC-emgnn_metrics.AUROC;
        rr.GCBRB_AUPRC = gc_metrics.AUPRC;
        rr.EMGNN_AUPRC = emgnn_metrics.AUPRC;
        rr.DeltaAUPRC = gc_metrics.AUPRC-emgnn_metrics.AUPRC;
        run_metric_rows = append_struct_local(run_metric_rows,rr);

        fprintf(['  Run %d <-> seed %d | genes=%d | ' ...
            'labels matched | order realigned\n'], ...
            run,seed,height(A));
    end

    gc_average = mean(gc_score_matrix,2);
    emgnn_average = mean(emgnn_score_matrix,2);

    averaged = table(reference_ensembl,reference_symbols, ...
        reference_keys,reference_labels,gc_average,emgnn_average, ...
        'VariableNames',{'EnsemblID','GeneSymbol','GeneKey','Label', ...
        'GCBRB_MeanProbability','EMGNN_MeanProbability'});

    writetable(averaged,fullfile(aligned_dir, ...
        sprintf('%s_aligned_average_predictions.csv',target)));

    gc_metrics = binary_metrics( ...
        reference_labels,gc_average,0.5);
    emgnn_metrics = binary_metrics( ...
        reference_labels,emgnn_average,0.5);

    delong = paired_delong_test( ...
        reference_labels,gc_average,emgnn_average);

    bootstrap_seed = cfg.bootstrap_seed + 1000*target_i;
    boot = paired_stratified_bootstrap_auprc( ...
        reference_labels,gc_average,emgnn_average, ...
        cfg.bootstrap_resamples,cfg.confidence_level, ...
        bootstrap_seed);

    row = struct();
    row.Network = string(target);
    row.NumTestGenes = numel(reference_labels);
    row.NumPositive = nnz(reference_labels==1);
    row.NumNegative = nnz(reference_labels==0);
    row.GCBRB_AUROC = gc_metrics.AUROC;
    row.EMGNN_AUROC = emgnn_metrics.AUROC;
    row.DeltaAUROC = gc_metrics.AUROC-emgnn_metrics.AUROC;
    row.DeLongZ = delong.Z;
    row.DeLongP_Raw = delong.PValue;
    row.GCBRB_AUPRC = gc_metrics.AUPRC;
    row.EMGNN_AUPRC = emgnn_metrics.AUPRC;
    row.DeltaAUPRC = gc_metrics.AUPRC-emgnn_metrics.AUPRC;
    row.AUPRC_Delta_CI_Lower = boot.CILower;
    row.AUPRC_Delta_CI_Upper = boot.CIUpper;
    row.BootstrapP_Raw = boot.PValue;
    row.BootstrapResamples = cfg.bootstrap_resamples;

    result_rows = append_struct_local(result_rows,row);

    fprintf('  Mean-prediction delta AUROC = %+.4f | raw p=%.6g\n', ...
        row.DeltaAUROC,row.DeLongP_Raw);
    fprintf(['  Mean-prediction delta AUPRC = %+.4f | ' ...
        '95%% CI=[%.4f, %.4f] | raw p=%.6g\n'], ...
        row.DeltaAUPRC,row.AUPRC_Delta_CI_Lower, ...
        row.AUPRC_Delta_CI_Upper,row.BootstrapP_Raw);
end

R = struct2table(result_rows);

R.DeLongP_Holm = holm_adjust_pvalues(R.DeLongP_Raw);
R.BootstrapP_Holm = holm_adjust_pvalues(R.BootstrapP_Raw);

R.AUROC_Significant_Holm = ...
    R.DeLongP_Holm < cfg.significance_level;
R.AUPRC_Significant_Holm = ...
    R.BootstrapP_Holm < cfg.significance_level;

conclusion = strings(height(R),1);
for i = 1:height(R)
    auc_sig = R.AUROC_Significant_Holm(i);
    ap_sig = R.AUPRC_Significant_Holm(i);

    if auc_sig && ap_sig
        conclusion(i) = "AUROC and AUPRC significant after Holm correction";
    elseif ap_sig
        conclusion(i) = "AUPRC significant; AUROC not significant after Holm correction";
    elseif auc_sig
        conclusion(i) = "AUROC significant; AUPRC not significant after Holm correction";
    else
        conclusion(i) = "Neither AUROC nor AUPRC significant after Holm correction";
    end
end
R.Conclusion = conclusion;

writetable(struct2table(audit_rows), ...
    fullfile(cfg.output_dir,'alignment_audit_30_pairs.csv'));
writetable(struct2table(run_metric_rows), ...
    fullfile(cfg.output_dir,'run_level_metrics_30_pairs.csv'));
writetable(R,fullfile(cfg.output_dir, ...
    'final_gcbrb_vs_emgnn_statistics.csv'));

paper_table = R(:,{ ...
    'Network','DeltaAUROC','DeLongP_Holm', ...
    'DeltaAUPRC','AUPRC_Delta_CI_Lower', ...
    'AUPRC_Delta_CI_Upper','BootstrapP_Holm','Conclusion'});

writetable(paper_table,fullfile(cfg.output_dir, ...
    'paper_statistical_table.csv'));

fprintf('\n================ FINAL STATISTICAL TABLE ================\n');
disp(paper_table);

fprintf('\nCompleted. Results saved in:\n  %s\n',cfg.output_dir);
fprintf(['Use paper_statistical_table.csv for the manuscript table, ' ...
    'after rounding values consistently.\n']);
end

function emgnn_dir = resolve_emgnn_table1_dir_local(cfg)
% Resolve outputs/table1 automatically or ask the user once.

path_record = fullfile(cfg.project_root,'EMGNN_TABLE1_PATH.txt');

candidates = strings(0,1);

if isfield(cfg,'emgnn_table1_dir') ...
        && strlength(string(cfg.emgnn_table1_dir)) > 0
    candidates(end+1,1) = string(cfg.emgnn_table1_dir);
end

if isfile(path_record)
    saved = strtrim(string(fileread(path_record)));
    if strlength(saved) > 0
        candidates(end+1,1) = saved;
    end
end

parent1 = fileparts(cfg.project_root);
parent2 = fileparts(parent1);

candidates = [candidates
    string(fullfile(cfg.project_root,'outputs','table1'))
    string(fullfile(cfg.project_root,'table1'))
    string(fullfile(parent1,'outputs','table1'))
    string(fullfile(parent1,'table1'))
    string(fullfile(parent2,'outputs','table1'))
    string(fullfile(parent2,'table1'))];

for i = 1:numel(candidates)
    candidate = char(candidates(i));
    candidate = normalize_selected_emgnn_dir_local(candidate);
    if is_valid_emgnn_dir_local(candidate,cfg)
        emgnn_dir = candidate;
        write_text_local(path_record,emgnn_dir);
        return;
    end
end

selected = uigetdir(cfg.project_root, ...
    'Select the EMGNN table1 folder or its outputs folder');

if isequal(selected,0)
    error('No EMGNN prediction directory was selected.');
end

selected = normalize_selected_emgnn_dir_local(selected);

if ~is_valid_emgnn_dir_local(selected,cfg)
    error(['The selected folder does not contain the expected EMGNN ' ...
        'prediction folders:\n  %s'],selected);
end

emgnn_dir = selected;
write_text_local(path_record,emgnn_dir);
end

function selected = normalize_selected_emgnn_dir_local(selected)
if isempty(selected)
    return;
end

if isfolder(fullfile(selected,'table1'))
    selected = fullfile(selected,'table1');
elseif isfolder(fullfile(selected,'outputs','table1'))
    selected = fullfile(selected,'outputs','table1');
end
end

function tf = is_valid_emgnn_dir_local(path,cfg)
tf = false;
if ~isfolder(path)
    return;
end

probe = fullfile(path, ...
    sprintf('%s_seed%d',cfg.emgnn_prefixes{1}, ...
    cfg.emgnn_seeds(1)),'test_predictions.csv');

tf = isfile(probe);
end

function G = read_gc_prediction_local(path)
T = readtable(path,'TextType','string', ...
    'VariableNamingRule','preserve');

id = get_column_local(T,{'EnsemblID'});
symbol = get_column_local(T,{'GeneSymbol'});
label = get_column_local(T,{'Label','Truth'});
score = get_column_local(T,{'FinalProbability','FinalScore'});

G = table(normalize_ensembl_local(id), ...
    normalize_symbol_local(symbol),double(label),double(score), ...
    'VariableNames',{'EnsemblID','GeneSymbol','Label','Probability'});
end

function E = read_emgnn_prediction_local(path)
T = readtable(path,'TextType','string', ...
    'VariableNamingRule','preserve');

id = get_column_local(T,{'entrez_id','ensembl_id','EnsemblID'});
symbol = get_column_local(T,{'gene_symbol','GeneSymbol'});
label = get_column_local(T,{'label','Label','Truth'});
score = get_column_local(T, ...
    {'probability_cancer','probability','score','FinalProbability'});

E = table(normalize_ensembl_local(id), ...
    normalize_symbol_local(symbol),double(label),double(score), ...
    'VariableNames',{'EnsemblID','GeneSymbol','Label','Probability'});
end

function value = get_column_local(T,candidates)
names = string(T.Properties.VariableNames);

for i = 1:numel(candidates)
    idx = find(strcmpi(names,string(candidates{i})),1);
    if ~isempty(idx)
        value = T.(T.Properties.VariableNames{idx});
        return;
    end
end

error('None of the required columns were found: %s', ...
    strjoin(string(candidates),', '));
end

function id = normalize_ensembl_local(id)
% Preserve a usable identifier even when MATLAB imports empty text as
% <missing>. Ensembl version suffixes are removed for cross-file matching.
id = string(id);
id(ismissing(id)) = "";
id = upper(strtrim(id));
id = regexprep(id,'\.\d+$','');
end

function symbol = normalize_symbol_local(symbol)
% Gene symbol is secondary identity information. Empty values are allowed
% because some benchmark rows contain an Ensembl ID without a symbol.
symbol = string(symbol);
symbol(ismissing(symbol)) = "";
symbol = upper(strtrim(symbol));
end

function key = build_alignment_key_local(ensembl_id,gene_symbol)
% Use stable Ensembl identity first. Fall back to gene symbol only when the
% Ensembl ID is absent. This avoids losing a valid gene solely because its
% symbol field is blank.
ensembl_id = string(ensembl_id(:));
gene_symbol = string(gene_symbol(:));

key = strings(numel(ensembl_id),1);
has_ensembl = strlength(ensembl_id) > 0;
has_symbol = strlength(gene_symbol) > 0;

key(has_ensembl) = "ENSG:" + ensembl_id(has_ensembl);
fallback = ~has_ensembl & has_symbol;
key(fallback) = "SYMBOL:" + gene_symbol(fallback);

invalid = ~has_ensembl & ~has_symbol;
if any(invalid)
    bad_rows = find(invalid);
    preview = strjoin(string(bad_rows(1:min(10,end))).',', ');
    error(['Found %d row(s) with neither Ensembl ID nor gene symbol. ' ...
        'First row indices: %s'],nnz(invalid),preview);
end
end

function [A,audit] = align_prediction_pair_local( ...
    G,E,target,run,seed,cfg)

G.Key = build_alignment_key_local(G.EnsemblID,G.GeneSymbol);
E.Key = build_alignment_key_local(E.EnsemblID,E.GeneSymbol);

if numel(unique(G.Key)) ~= height(G)
    error(['Duplicate GC-BRB alignment keys in %s run %d. ' ...
        'Check duplicated Ensembl IDs.'],target,run);
end
if numel(unique(E.Key)) ~= height(E)
    error(['Duplicate EMGNN alignment keys in %s seed %d. ' ...
        'Check duplicated Ensembl IDs.'],target,seed);
end

[common_key,ig,ie] = intersect(G.Key,E.Key,'sorted');

if cfg.require_exact_gene_set
    if numel(common_key) ~= height(G) || numel(common_key) ~= height(E)
        gc_only = setdiff(G.Key,E.Key);
        emgnn_only = setdiff(E.Key,G.Key);

        gc_preview = strjoin(gc_only(1:min(5,numel(gc_only))),', ');
        emgnn_preview = strjoin(emgnn_only(1:min(5,numel(emgnn_only))),', ');

        error(['Gene-set mismatch for %s run %d / EMGNN seed %d. ' ...
            'GC=%d, EMGNN=%d, common=%d. ' ...
            'GC-only examples: %s. EMGNN-only examples: %s.'], ...
            target,run,seed,height(G),height(E),numel(common_key), ...
            gc_preview,emgnn_preview);
    end
end

g_label = G.Label(ig);
e_label = E.Label(ie);

if cfg.require_exact_labels && ~isequal(g_label,e_label)
    mismatch = nnz(g_label~=e_label);
    error(['Truth-label mismatch for %s run %d / seed %d: ' ...
        '%d genes.'],target,run,seed,mismatch);
end

g_symbol = G.GeneSymbol(ig);
e_symbol = E.GeneSymbol(ie);
both_symbols = strlength(g_symbol)>0 & strlength(e_symbol)>0;
symbol_mismatch = both_symbols & g_symbol~=e_symbol;

% Prefer the GC-BRB symbol. Fill a blank symbol from EMGNN only for export.
export_symbol = g_symbol;
fill_from_emgnn = strlength(export_symbol)==0 & strlength(e_symbol)>0;
export_symbol(fill_from_emgnn) = e_symbol(fill_from_emgnn);

export_ensembl = G.EnsemblID(ig);
fill_ensembl = strlength(export_ensembl)==0 & ...
    strlength(E.EnsemblID(ie))>0;
export_ensembl(fill_ensembl) = E.EnsemblID(ie(fill_ensembl));

A = table(common_key,export_ensembl,export_symbol,g_label, ...
    G.Probability(ig),E.Probability(ie), ...
    'VariableNames',{'Key','EnsemblID','GeneSymbol','Label', ...
    'GCBRBProbability','EMGNNProbability'});

audit = struct();
audit.Network = string(target);
audit.GCBRBRun = run;
audit.EMGNNSeed = seed;
audit.IdentityMode = "Ensembl-first; symbol fallback";
audit.GCBRBRows = height(G);
audit.EMGNNRows = height(E);
audit.CommonRows = height(A);
audit.ExactGeneSet = ...
    height(A)==height(G) && height(A)==height(E);
audit.LabelMismatchCount = nnz(g_label~=e_label);
audit.GCBRBBlankEnsemblCount = nnz(strlength(G.EnsemblID)==0);
audit.GCBRBBlankSymbolCount = nnz(strlength(G.GeneSymbol)==0);
audit.EMGNNBlankEnsemblCount = nnz(strlength(E.EnsemblID)==0);
audit.EMGNNBlankSymbolCount = nnz(strlength(E.GeneSymbol)==0);
audit.NonemptySymbolMismatchCount = nnz(symbol_mismatch);
audit.AlignmentStatus = "PASS";
end


function require_file_local(path,label)
if ~isfile(path)
    error('%s was not found:\n  %s',label,path);
end
end

function write_text_local(path,text)
fid = fopen(path,'w');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s',text);
end

function output = append_struct_local(output,row)
if isempty(output)
    output = row;
else
    row = orderfields(row,output);
    output(end+1,1) = row;
end
end
