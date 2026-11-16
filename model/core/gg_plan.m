function plan=gg_plan(data,id,split,cfg,folder)
if ~isfolder(folder),mkdir(folder);end
outer=gg_folds(split.train_idx,id,data.y,cfg.outer_folds,split.seed+210001);
plans=cell(cfg.outer_folds+1,1);fold_id=zeros(numel(data.y),1);
for k=1:numel(outer)
    held=outer{k};pool=setdiff(split.train_idx,held,'stable');
    seed=split.seed+1000000*k;
    [fit,stop]=gg_holdout(pool,id,data.y,cfg.validation_fraction,seed+220001);
    plans{k}=struct('name',sprintf('outer_%02d',k),'seed',seed, ...
        'fit_idx',fit,'stop_idx',stop,'held_idx',held);
    fold_id(held)=k;
end
plans{end}=struct('name','final','seed',split.seed, ...
    'fit_idx',split.train_idx,'stop_idx',split.val_idx, ...
    'held_idx',split.test_idx);
rows=struct([]);
membership=table();
for j=1:numel(plans)
    p=plans{j};gg_assert_separate(id,p.fit_idx,p.stop_idx,p.held_idx,p.name);
    pi=[p.fit_idx;p.stop_idx;p.held_idx];
    role=[repmat("Fit",numel(p.fit_idx),1);repmat("Stop",numel(p.stop_idx),1); ...
        repmat("Held",numel(p.held_idx),1)];
    P=table(repmat(string(p.name),numel(pi),1),pi,id.group_id(pi), ...
        role,double(data.y(pi)),'VariableNames', ...
        {'Pipeline','GlobalIndex','GroupID','Role','Label'});
    if isempty(membership),membership=P;else,membership=[membership;P];end %#ok<AGROW>
    row=struct('Pipeline',string(p.name),'Check',"fit_stop_held", ...
        'FitRecords',numel(p.fit_idx),'HeldRecords',numel(p.held_idx), ...
        'OverlappingGroups',0,'Passed',true);rows=append_struct(rows,row);
    full=gg_folds(p.fit_idx,id,data.y,cfg.omics_tuning_folds,p.seed+120001);
    for k=1:numel(full)
        fit=setdiff(p.fit_idx,full{k},'stable');
        gg_assert_separate(id,fit,full{k},[],'full omics tuning');
    end
    inner=gg_folds(p.fit_idx,id,data.y,cfg.crossfit_folds,p.seed+130001);
    for k=1:numel(inner)
        seeds=setdiff(p.fit_idx,inner{k},'stable');
        gg_assert_separate(id,seeds,inner{k},p.held_idx,'inner evidence');
        nested=gg_folds(seeds,id,data.y,cfg.omics_tuning_folds,p.seed+140000+k);
        for q=1:numel(nested)
            gg_assert_separate(id,setdiff(seeds,nested{q},'stable'), ...
                nested{q},inner{k},'nested omics tuning');
        end
        row.Check=string(sprintf('evidence_fold_%d_and_nested_tuning',k));
        row.FitRecords=numel(seeds);row.HeldRecords=numel(inner{k});
        rows=append_struct(rows,row);
    end
end
part=repmat("",numel(data.y),1);
part(split.train_idx)="Train";part(split.val_idx)="Validation";
part(split.test_idx)="Test";part(split.excluded_idx)="ExcludedTestAlias";
use=find(part~="");T=id.record_table(use,:);T.Partition=part(use);
T.OuterFold=fold_id(use);
writetable(T,fullfile(folder,'split_manifest.csv'));
writetable(membership,fullfile(folder,'pipeline_memberships.csv'));
names={'train','validation','test'};
indices={split.train_idx,split.val_idx,split.test_idx};
for j=1:3
    ix=indices{j};
    M=table(ix,ix-1,double(data.y(ix)),id.group_id(ix), ...
        string(data.ensembl_ids(ix)),string(data.gene_symbols(ix)), ...
        'VariableNames',{'GlobalIndex_1based','GlobalIndex_0based','Label', ...
        'GroupID','EnsemblID','GeneSymbol'});
    writetable(M,fullfile(folder,[names{j} '_identity.csv']));
end
writetable(struct2table(rows),fullfile(folder,'safety_checks.csv'));
orig=readtable(fullfile(cfg.project_root,'reference','expected_test_counts.csv'));
ref=orig(strcmp(string(orig.Target),string(split.target)),:);
if numel(split.excluded_idx)~=ref.ExcludedLabelledRecordN ...
        || numel(unique(id.group_id(split.test_idx)))~=ref.TestGroupN
    error('GG:Reference','Independent exclusion-count check failed.');
end
plan=struct('pipelines',{plans},'outer_fold',fold_id,'split',split);
gg_save(fullfile(folder,'plan.mat'),struct('plan',plan));
end
