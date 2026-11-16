function split = gg_split(data,id,target,run,cfg)
target_i=find(strcmpi(cfg.all_datasets,target),1);
layer_i=find(strcmpi(data.layer_names,target),1);
if isempty(target_i)||isempty(layer_i), error('GG:Target','Unknown target.'); end
seed=cfg.seed_base+1000*target_i+run;
ref=readtable(fullfile(cfg.project_root,'reference','frozen_test', ...
    [target '.csv']));
test_idx=double(ref.GlobalIndex_1based(:));
expected=unique(double(data.layers{layer_i}.test_global(:)));
if ~isequal(sort(test_idx),sort(expected)) ...
        || any(data.y(test_idx)~=double(ref.Label(:)))
    error('GG:Reference','Prepared mapping does not match frozen %s test.',target);
end
labelled=find(isfinite(data.y));
test_groups=unique(id.group_id(test_idx));
official_pool=setdiff(labelled,test_idx,'stable');
exclude_flag=ismember(id.group_id(official_pool),test_groups);
excluded_idx=official_pool(exclude_flag);
pool=official_pool(~exclude_flag);
[train_idx,val_idx]=gg_holdout(pool,id,data.y,cfg.validation_fraction,seed);
gg_assert_separate(id,train_idx,val_idx,test_idx,'top train/validation/test');
if ~isequal(sort([train_idx;val_idx;test_idx;excluded_idx]),sort(labelled))
    error('GG:Coverage','Split accounting is incomplete.');
end
split=struct('target',target,'run',run,'seed',seed, ...
    'target_layer_index',layer_i,'train_idx',train_idx, ...
    'val_idx',val_idx,'test_idx',test_idx,'excluded_idx',excluded_idx);
end
