function gg_selftest()
%GG_SELFTEST Small numerical checks; no original data or results are written.
assert(strcmp(gg_hash('abc'), ...
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'));
bridge=struct('ensembl_ids',{{'ENSG1';'ENSG1';'ENSG2';'ENSG3'}}, ...
    'gene_symbols',{{'alpha';'BETA';'beta';'gamma'}},'y',[0;0;0;1]);
b=gg_identity(bridge);
assert(all(b.group_id(1:3)==b.group_id(1)) && b.group_id(4)~=b.group_id(1));
bridge.y(2)=1;conflict=gg_identity(bridge);assert(any(conflict.label_conflict));
data=synthetic_data();id=gg_identity(data);
cfg=gg_config('smoke');cfg.verbose=false;cfg.crossfit_verbose=false;
cfg.crossfit_folds=2;cfg.omics_tuning_folds=2;
cfg.omics_lambda_grid=[1e-4 1e-2];cfg.lr_lambda_grid=[1e-4 1e-2];
cfg.max_generations=2;cfg.rwr_max_iterations=15;
fit=find(id.group_id<=20);stop=find(id.group_id>20 & id.group_id<=24);
held=find(id.group_id>24);
gg_assert_separate(id,fit,stop,held,'synthetic');
[a,v]=gg_holdout(fit,id,data.y,0.2,102);
assert(isempty(intersect(id.group_id(a),id.group_id(v))));
folds=gg_folds(fit,id,data.y,3,103);
assert(isequal(sort(vertcat(folds{:})),sort(fit)));
failed=false;
try
    gg_assert_separate(id,fit,[stop;fit(1)],held,'intentional overlap');
catch ex
    failed=strcmp(ex.identifier,'GG:Overlap');
end
assert(failed,'Intentional group overlap was not rejected.');
folder=tempname;mkdir(folder);
cleanup=onCleanup(@()rmdir(folder,'s')); %#ok<NASGU>
one=gg_pipeline(data,id,fit,stop,held,cfg,104,fullfile(folder,'one'),'selftest');
changed=data;changed.y(held)=1-changed.y(held);
two=gg_pipeline(changed,id,fit,stop,held,cfg,104,fullfile(folder,'two'),'selftest');
assert(max(abs(one.rule_score(held)-two.rule_score(held)))<1e-10, ...
    'Changing forbidden labels changed BRB predictions.');
assert(max(abs(one.lr_score(held)-two.lr_score(held)))<1e-10, ...
    'Changing forbidden labels changed LR predictions.');
assert(max(abs(one.evidence.feature_matrix(:)-two.evidence.feature_matrix(:)))<1e-10, ...
    'Changing forbidden labels changed supervised evidence.');
assert(max(abs(one.brb.x-two.brb.x))<1e-10);
assert(one.lr.selected_lambda==two.lr.selected_lambda);
fprintf('GG SELFTEST PASSED: identity, grouping, overlap rejection, held-label invariance.\n');
end
function data=synthetic_data()
rng(101,'twister');N=60;S=6;
data=struct();data.y=repelem(mod((1:30).',2),2);
data.X=randn(N,64);data.num_global_genes=N;data.num_layers=S;
data.ensembl_ids=cell(N,1);data.gene_symbols=cell(N,1);
for j=1:30
    data.ensembl_ids{2*j-1}=sprintf('ENSG%011d',j);
    data.ensembl_ids{2*j}=sprintf('ENSG%011d',j);
    data.gene_symbols{2*j-1}=sprintf('GENE%03d',j);
    data.gene_symbols{2*j}=sprintf('gene%03d',j);
end
data.layer_names={'CPDB';'MULTINET';'PCNET';'STRINGdb';'IREF';'IREF_2015'};
data.presence=true(N,S);data.layers=cell(S,1);
for s=1:S
    a=(1:N).';b=mod(a,N)+1;c=mod(a+s,N)+1;
    A=spones(sparse([a;b;a;c],[b;a;c;a],1,N,N));
    A=A-spdiags(diag(A),0,N,N);
    data.layers{s}=struct('name',data.layer_names{s},'num_genes',N, ...
        'local_to_global',(1:N).','A',A,'degree',full(sum(A,2)));
end
end
