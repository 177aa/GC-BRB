function out = gg_pipeline(data,id,fit_idx,stop_idx,held_idx,cfg,seed,folder,base)
%GG_PIPELINE No held_idx label enters fitting, evidence or stopping.
if ~isfolder(folder),mkdir(folder);end
gg_assert_separate(id,fit_idx,stop_idx,held_idx,'complete pipeline');
key=struct('base',base,'seed',seed,'fit',fit_idx(:).', ...
    'stop',stop_idx(:).','held',held_idx(:).');
signature=gg_hash(jsonencode(key));
path=fullfile(folder,'pipeline.mat');
if cfg.resume && isfile(path)
    S=load(path,'out');
    if ~isfield(S,'out')||~strcmp(S.out.signature,signature)
        error('GG:Cache','Pipeline fingerprint differs: %s',path);
    end
    out=S.out;return;
end
safe=data;safe.y=nan(size(data.y));
safe.y(fit_idx)=data.y(fit_idx);safe.y(stop_idx)=data.y(stop_idx);
e_path=fullfile(folder,'evidence.mat');
if cfg.resume && isfile(e_path)
    E=load(e_path,'e','signature');
    if ~isfield(E,'signature')||~strcmp(E.signature,signature)
        error('GG:Cache','Evidence fingerprint differs: %s',e_path);
    end
    e=E.e;
else
    e=gg_build_evidence(safe,id,fit_idx,cfg,seed);
    gg_save(e_path,struct('e',e,'signature',signature));
end
F=table(fit_idx(:),id.group_id(fit_idx),e.crossfit_fold(fit_idx), ...
    'VariableNames',{'GlobalIndex','GroupID','EvidenceFold'});
writetable(F,fullfile(folder,'evidence_training_folds.csv'));
brb=gg_fit_brb(safe,id,e,fit_idx,stop_idx,cfg,seed,folder,signature);
lr=gg_fit_lr(e.feature_matrix,safe,id,fit_idx,stop_idx,cfg);
rule_score=multilayer_brb_forward_v3layer(brb.x,e,brb.model,[]);
out=struct('signature',signature,'fit_idx',fit_idx, ...
    'stop_idx',stop_idx,'held_idx',held_idx,'seed',seed,'evidence',e, ...
    'brb',brb,'lr',lr,'rule_score',rule_score,'lr_score',lr.score);
gg_save(path,struct('out',out));
writetable(e.omics_lambda_cv,fullfile(folder,'omics_full_lambda_selection.csv'));
writetable(lr.lambda_table,fullfile(folder,'lr_lambda_selection.csv'));
end
