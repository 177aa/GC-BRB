function gg_assert_separate(id,a,b,c,context)
if nargin<5, context='split'; end
sets={a(:),b(:),c(:)};
for j=1:3
    if numel(sets{j})~=numel(unique(sets{j}))
        error('GG:Duplicate','Duplicate indices in %s.',context);
    end
    for k=j+1:3
        if ~isempty(intersect(id.group_id(sets{j}),id.group_id(sets{k})))
            error('GG:Overlap','Identity groups cross sets in %s.',context);
        end
    end
end
end
