function selection=gg_select_alpha(y,rule,linear,idx,id,cfg)
%GG_SELECT_ALPHA Both branches must originate from the same outer folds.
[yy,r]=gg_group_scores(y,rule(idx),idx,id);
[~,l]=gg_group_scores(y,linear(idx),idx,id);
grid=cfg.fusion_alpha_grid(:);ap=nan(size(grid));brier=ap;
for j=1:numel(grid)
    p=(1-grid(j))*r+grid(j)*l;m=binary_metrics(yy,p,0.5);
    ap(j)=m.AUPRC;brier(j)=m.Brier;
end
best=max(ap);candidates=find(abs(ap-best)<=1e-12);
best_brier=min(brier(candidates));
candidates=candidates(abs(brier(candidates)-best_brier)<=1e-12);
[~,order]=min(grid(candidates));chosen=candidates(order);
selection=struct('alpha',grid(chosen),'rule_weight',1-grid(chosen), ...
    'lr_weight',grid(chosen),'unit','conservative_identity_group', ...
    'source','complete_grouped_outer_oof');
selection.grid=table(grid,ap,brier,(1:numel(grid)).'==chosen, ...
    'VariableNames',{'LRWeight','GroupOOFAUPRC','GroupOOFBrier','Selected'});
end
