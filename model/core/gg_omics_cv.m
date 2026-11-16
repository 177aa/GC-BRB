function [lambda,T] = gg_omics_cv(data,id,idx,cfg,seed)
%GG_OMICS_CV All tuning observations belong to this fitting subset.
folds=gg_folds(idx,id,data.y,cfg.omics_tuning_folds,seed);
grid=cfg.omics_lambda_grid(:);
ap=nan(size(grid)); brier=ap;
for j=1:numel(grid)
    oof=nan(numel(data.y),1);
    for k=1:numel(folds)
        held=folds{k}; fit=setdiff(idx,held,'stable');
        [~,p]=fit_omics_evidence(data.X,data.y,fit,cfg,grid(j));
        oof(held)=p(held);
    end
    [yy,pp]=gg_group_scores(data.y,oof(idx),idx,id);
    m=binary_metrics(yy,pp,0.5); ap(j)=m.AUPRC; brier(j)=m.Brier;
end
order=sortrows([-ap,brier,-grid,(1:numel(grid)).'],[1 2 3]);
selected=order(1,4); lambda=grid(selected);
T=table(grid,ap,brier,(1:numel(grid)).'==selected, ...
    'VariableNames',{'Lambda','GroupOOFAUPRC','GroupOOFBrier','Selected'});
end
