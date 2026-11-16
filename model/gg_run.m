function gg_run(stage)
gg_setup();cfg=gg_config(stage);
try
    gg_run_local(cfg);
catch ex
    if ~isfolder(cfg.result_root),mkdir(cfg.result_root);end
    gg_jsonwrite(fullfile(cfg.result_root,'LAST_FAILURE.json'), ...
        struct('Stage',cfg.stage,'Identifier',ex.identifier, ...
        'Message',ex.message,'Stack',ex.stack));
    rethrow(ex);
end
end
function gg_run_local(cfg)
stage=cfg.stage;
[data,id,base]=gg_load_data(cfg);
fprintf('Protocol %s | stage %s | conservative groups %d\n', ...
    cfg.protocol_version,cfg.stage,height(id.audit));
preflight_rows=struct([]);
for t=1:numel(cfg.target_datasets)
    target=cfg.target_datasets{t};
    for run=1:cfg.num_runs
        split=gg_split(data,id,target,run,cfg);
        folder=fullfile(cfg.result_root,target,sprintf('run_%02d',run));
        plan=gg_plan(data,id,split,cfg,folder);
        row=struct('Target',string(target),'Run',run,'Seed',split.seed, ...
            'TrainRecords',numel(split.train_idx), ...
            'ValidationRecords',numel(split.val_idx),'TestRecords',numel(split.test_idx), ...
            'ExcludedLabelledAliases',numel(split.excluded_idx), ...
            'TrainGroups',numel(unique(id.group_id(split.train_idx))), ...
            'ValidationGroups',numel(unique(id.group_id(split.val_idx))), ...
            'TestGroups',numel(unique(id.group_id(split.test_idx))), ...
            'AllPlannedGroupIsolationChecksPassed',true);
        preflight_rows=append_struct(preflight_rows,row);
        writetable(struct2table(preflight_rows), ...
            fullfile(cfg.result_root,'preflight_summary.csv'));
        if strcmp(stage,'preflight'),continue;end
        fprintf('\n===== %s run %d: full outer OOF =====\n',target,run);
        result_path=fullfile(folder,'run_result.mat');
        if cfg.resume && isfile(result_path)
            S=load(result_path,'result');
            if ~strcmp(S.result.base_fingerprint,base)
                error('GG:Cache','Run fingerprint differs: %s',folder);
            end
            fprintf('Completed run found; collecting its saved result.\n');
            continue;
        end
        oof_rule=nan(numel(data.y),1);oof_lr=oof_rule;
        for k=1:cfg.outer_folds
            p=plan.pipelines{k};
            out=gg_pipeline(data,id,p.fit_idx,p.stop_idx,p.held_idx,cfg, ...
                p.seed,fullfile(folder,p.name),base);
            oof_rule(p.held_idx)=out.rule_score(p.held_idx);
            oof_lr(p.held_idx)=out.lr_score(p.held_idx);
        end
        if any(~isfinite(oof_rule(split.train_idx))) ...
                || any(~isfinite(oof_lr(split.train_idx)))
            error('GG:OOF','Incomplete outer predictions.');
        end
        selection=gg_select_alpha(data.y,oof_rule,oof_lr,split.train_idx,id,cfg);
        writetable(selection.grid,fullfile(folder,'fusion_alpha_outer_oof.csv'));
        ix=split.train_idx;
        O=table(ix,id.group_id(ix),data.y(ix),plan.outer_fold(ix), ...
            oof_rule(ix),oof_lr(ix),'VariableNames',{'GlobalIndex','GroupID', ...
            'Label','OuterFold','RuleOOF','LROOF'});
        writetable(O,fullfile(folder,'outer_oof_predictions.csv'));
        p=plan.pipelines{end};
        final=gg_pipeline(data,id,p.fit_idx,p.stop_idx,p.held_idx,cfg, ...
            p.seed,fullfile(folder,'final'),base);
        result=gg_export_run(data,id,split,final,selection, ...
            oof_rule,oof_lr,cfg,folder,base);
        gg_save(result_path,struct('result',result));
        fprintf('Completed %s run %d: group test AP %.6f; alpha %.2f\n', ...
            target,run,result.summary.GroupTestAUPRC,selection.alpha);
        gg_collect(cfg);
    end
end
if strcmp(stage,'preflight')
    gg_jsonwrite(fullfile(cfg.result_root,'PREFLIGHT_PASSED.json'), ...
        struct('Passed',true,'RunsAudited',numel(preflight_rows), ...
        'BaseFingerprint',base,'ModelTrainingPerformed',false));
    fprintf('All %d planned splits passed. No model was trained.\n',numel(preflight_rows));
else
    gg_collect(cfg);
end
end
