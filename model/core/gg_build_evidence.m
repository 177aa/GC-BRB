function e = gg_build_evidence(data,id,fit_idx,cfg,seed)
%GG_BUILD_EVIDENCE Group OOF evidence with fold-local omics tuning.
% No labels outside fit_idx are available to this function's computations.
safe=data; safe.y=nan(size(data.y));safe.y(fit_idx)=data.y(fit_idx);
[lambda,cv]=gg_omics_cv(safe,id,fit_idx,cfg,seed+120001);
[omics_model,omics,coeff]=fit_omics_evidence( ...
    safe.X,safe.y,fit_idx,cfg,lambda);
top=compute_topology_evidence(safe,fit_idx,cfg,'Full ');
e=struct('omics_score',omics,'onehop',top.onehop,'twohop',top.twohop, ...
    'diffusion',top.diffusion,'confidence',top.confidence, ...
    'presence',logical(data.presence),'omics_model',omics_model, ...
    'omics_selected_lambda',lambda,'omics_lambda_cv',cv, ...
    'omics_coefficients',coeff,'rwr_iterations',top.rwr_iterations);
folds=gg_folds(fit_idx,id,safe.y,cfg.crossfit_folds,seed+130001);
e.crossfit_fold=zeros(numel(data.y),1);
e.fold_selected_lambda=zeros(numel(folds),1);
e.fold_lambda_cv=cell(numel(folds),1);
e.seed_indices=cell(numel(folds),1);
for k=1:numel(folds)
    held=folds{k}; seeds=setdiff(fit_idx,held,'stable');
    gg_assert_separate(id,seeds,held,[],'evidence cross-fitting');
    folded=safe;folded.y(:)=NaN;folded.y(seeds)=safe.y(seeds);
    [fold_lambda,fold_cv]=gg_omics_cv(folded,id,seeds,cfg,seed+140000+k);
    [~,p]=fit_omics_evidence(folded.X,folded.y,seeds,cfg,fold_lambda);
    t=compute_topology_evidence(folded,seeds,cfg,sprintf('Inner%d ',k));
    e.omics_score(held)=p(held);
    names={'onehop','twohop','diffusion','confidence'};
    for j=1:numel(names), e.(names{j})(held,:)=t.(names{j})(held,:); end
    e.crossfit_fold(held)=k;
    e.fold_selected_lambda(k)=fold_lambda;e.fold_lambda_cv{k}=fold_cv;
    e.seed_indices{k}=seeds;
end
if any(e.crossfit_fold(fit_idx)==0),error('GG:OOF','Missing OOF evidence.');end
e.confidence(~e.presence)=0;
e.feature_matrix=gg_features(e);
e.version='GG_OOF_1_0_EVIDENCE';
end
