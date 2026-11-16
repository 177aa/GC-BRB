function brb = gg_fit_brb(data,id,e,fit_idx,stop_idx,cfg,seed,folder,signature)
%GG_FIT_BRB Stopping/restart selection uses only the internal stop subset.
split=struct('train_idx',fit_idx);
[x0,model]=generator_multilayer_brb_v3layer(data,e,split,cfg);
objective=@(x)multilayer_training_loss_v3layer(x,e,data,model,fit_idx,cfg);
validation=@(x)group_objective(x,e,data,id,model,stop_idx);
project=@(x)project_multilayer_parameters(x,model);
best=Inf;chosen=[];outputs=cell(cfg.cmaes_restarts,1);
xs=cell(cfg.cmaes_restarts,1);rows=struct([]);
for restart=1:cfg.cmaes_restarts
    path=fullfile(folder,sprintf('brb_restart_%02d.mat',restart));
    if cfg.resume && isfile(path)
        S=load(path);
        if ~isfield(S,'signature')||~strcmp(S.signature,signature)
            error('GG:Cache','Restart fingerprint differs: %s',path);
        end
        x=S.x;output=S.output;
    else
        rng(seed+100000*restart,'twister');
        options=struct('max_generations',cfg.max_generations, ...
            'sigma0',cfg.sigma0,'population_size',cfg.population_size, ...
            'patience',cfg.early_stop_patience,'verbose',cfg.verbose);
        [x,output]=cmaes(objective,x0,options,project,validation);
        gg_save(path,struct('x',x,'output',output,'signature',signature));
    end
    value=validation(x);xs{restart}=x;outputs{restart}=output;
    row=struct('Restart',restart,'StopGroupAUPRC',-value, ...
        'FitLoss',objective(x),'Generations',output.generations);
    rows=append_struct(rows,row);
    if value<best, best=value;chosen=x;selected=restart;end
end
if cfg.evaluate_parameter_consensus && numel(xs)>1
    consensus=project(mean(cat(2,xs{:}),2));value=validation(consensus);
    if value<best,chosen=consensus;best=value;selected=0;end
end
brb=struct('x',chosen,'model',model,'selected_restart',selected, ...
    'stop_group_auprc',-best,'restarts',struct2table(rows));
writetable(brb.restarts,fullfile(folder,'brb_restart_selection.csv'));
end
function v=group_objective(x,e,data,id,model,idx)
p=multilayer_brb_forward_v3layer(x,e,model,idx);
[yy,pp]=gg_group_scores(data.y,p,idx,id);
m=binary_metrics(yy,pp,0.5);v=-m.AUPRC;
if ~isfinite(v),error('GG:Validation','Invalid stopping objective.');end
end
