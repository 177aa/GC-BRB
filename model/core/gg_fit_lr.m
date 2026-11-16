function lr = gg_fit_lr(F,data,id,fit_idx,stop_idx,cfg)
%GG_FIT_LR Fit only on fit_idx; tune lambda on a disjoint stop subset.
% This independent-subset route avoids CV on a globally prebuilt matrix.
mu=mean(F(fit_idx,:),1);sigma=std(F(fit_idx,:),0,1);
sigma(sigma<1e-12)=1;
train=(F(fit_idx,:)-mu)./sigma;
y=double(data.y(fit_idx));
weights=zeros(size(y));weights(y==1)=0.5/nnz(y==1);
weights(y==0)=0.5/nnz(y==0);weights=weights*numel(y);
grid=cfg.lr_lambda_grid(:);models=cell(numel(grid),1);
ap=nan(size(grid));brier=ap;
for j=1:numel(grid)
    mdl=fitclinear(train,y,'Learner','logistic', ...
        'Regularization','ridge','Lambda',grid(j),'Solver','lbfgs', ...
        'Weights',weights,'ClassNames',[0;1],'Verbose',0);
    models{j}=mdl;
    p=lr_predict(mdl,F(stop_idx,:),mu,sigma);
    [yy,pp]=gg_group_scores(data.y,p,stop_idx,id);
    m=binary_metrics(yy,pp,0.5);ap(j)=m.AUPRC;brier(j)=m.Brier;
end
order=sortrows([-ap,brier,-grid,(1:numel(grid)).'],[1 2 3]);j=order(1,4);
lr=struct('classifier',models{j},'mu',mu,'sigma',sigma, ...
    'selected_lambda',grid(j),'selection_subset','independent_internal_stop');
lr.lambda_table=table(grid,ap,brier,(1:numel(grid)).'==j, ...
    'VariableNames',{'Lambda','StopGroupAUPRC','StopGroupBrier','Selected'});
lr.score=lr_predict(lr.classifier,F,mu,sigma);
end
function p=lr_predict(mdl,F,mu,sigma)
[~,raw]=predict(mdl,(F-mu)./sigma);
column=find(double(mdl.ClassNames)==1,1);p=double(raw(:,column));
if any(p<0|p>1),p=1./(1+exp(-min(max(p,-40),40)));end
p=min(max(p,0),1);
if any(~isfinite(p)),error('GG:LR','Nonfinite LR score.');end
end
