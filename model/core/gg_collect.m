function gg_collect(cfg)
rows=struct([]);
protocol=jsondecode(fileread(fullfile(cfg.result_root,'protocol.json')));
if strcmp(cfg.profile,'formal')
    collect_targets=cfg.all_datasets;collect_runs=5;
else
    collect_targets=cfg.target_datasets;collect_runs=cfg.num_runs;
end
for t=1:numel(collect_targets)
    target=collect_targets{t};
    for run=1:collect_runs
        path=fullfile(cfg.result_root,target,sprintf('run_%02d',run),'run_result.mat');
        if ~isfile(path),continue;end
        S=load(path,'result');
        if ~strcmp(S.result.base_fingerprint,protocol.BaseFingerprint)
            error('GG:Collect','Mixed fingerprints in %s.',path);
        end
        rows=append_struct(rows,S.result.summary);
    end
end
if isempty(rows),return;end
T=struct2table(rows);
writetable(T,fullfile(cfg.result_root,'all_run_results.csv'));
targets=unique(T.Target,'stable');summary=struct([]);
metrics={'GroupTestAUPRC','GroupTestAUROC','GroupTestMCC','GroupTestF1', ...
    'GroupTestBrier','TestAUPRC','TestAUROC'};
for k=1:numel(targets)
    part=T(T.Target==targets(k),:);
    row=struct('Target',targets(k),'CompletedRuns',height(part));
    for j=1:numel(metrics)
        values=part.(metrics{j});
        row.([metrics{j} '_Mean'])=mean(values);
        row.([metrics{j} '_Std'])=std(values);
    end
    summary=append_struct(summary,row);
end
S=struct2table(summary);writetable(S,fullfile(cfg.result_root,'summary_results.csv'));
expected=numel(collect_targets)*collect_runs;
complete=height(T)==expected;
gg_jsonwrite(fullfile(cfg.result_root,'completion.json'), ...
    struct('Stage',cfg.stage,'CompletedRuns',height(T),'ExpectedRuns',expected, ...
    'AllStudyRunsComplete',complete,'Full30RunStudyComplete', ...
    complete && strcmp(cfg.profile,'formal') && expected==30));
if complete
    macro=table(string(cfg.profile),height(S),height(T), ...
        mean(S.GroupTestAUPRC_Mean),mean(S.TestAUPRC_Mean), ...
        'VariableNames',{'Stage','Targets','Runs','GroupMacroAUPRC','EntryMacroAUPRC'});
    writetable(macro,fullfile(cfg.result_root,'macro_summary.csv'));
end
end
