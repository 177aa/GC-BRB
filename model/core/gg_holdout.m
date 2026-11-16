function [fit_idx,val_idx] = gg_holdout(indices,id,y,fraction,seed)
%GG_HOLDOUT Stratify GROUPS, never individual paired records.
indices=indices(:); [groups,first,g]=unique(id.group_id(indices));
labels=double(y(indices(first)));
if any(~ismember(labels,[0 1])) || any(double(y(indices))~=labels(g))
    error('GG:Labels','Holdout group has missing or conflicting labels.');
end
stream=RandStream('mt19937ar','Seed',seed);
val_groups=[];
for c=0:1
    pool=groups(labels==c);
    if numel(pool)<2, error('GG:Small','Need two groups per class.'); end
    pool=pool(randperm(stream,numel(pool)));
    n=max(1,min(numel(pool)-1,round(fraction*numel(pool))));
    val_groups=[val_groups;pool(1:n)]; %#ok<AGROW>
end
flag=ismember(id.group_id(indices),val_groups);
fit_idx=indices(~flag); val_idx=indices(flag);
gg_assert_separate(id,fit_idx,val_idx,[],'grouped holdout');
end
