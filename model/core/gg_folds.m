function folds = gg_folds(indices,id,y,K,seed)
indices=indices(:); [groups,first,g]=unique(id.group_id(indices));
labels=double(y(indices(first)));
if any(~ismember(labels,[0 1])) || any(double(y(indices))~=labels(g))
    error('GG:Labels','Fold group has missing or conflicting labels.');
end
stream=RandStream('mt19937ar','Seed',seed);
held_groups=cell(K,1);
for c=0:1
    pool=groups(labels==c);
    if numel(pool)<K
        error('GG:Small','Need at least %d groups per class.',K);
    end
    pool=pool(randperm(stream,numel(pool)));
    for k=1:K, held_groups{k}=[held_groups{k};pool(k:K:end)]; end
end
folds=cell(K,1);
for k=1:K
    folds{k}=indices(ismember(id.group_id(indices),held_groups{k}));
    gg_assert_separate(id,setdiff(indices,folds{k},'stable'), ...
        folds{k},[],'grouped fold');
end
if ~isequal(sort(vertcat(folds{:})),sort(indices))
    error('GG:Folds','Folds lost or duplicated records.');
end
end
