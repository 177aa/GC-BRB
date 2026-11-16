function [truth,score,group_id] = gg_group_scores(y,p,indices,id)
indices=indices(:); p=double(p(:));
if numel(p)~=numel(indices), error('GG:Length','Score length mismatch.'); end
[group_id,~,g]=unique(id.group_id(indices));
score=accumarray(g,p,[],@mean);
truth=accumarray(g,double(y(indices)),[],@mean);
if any(~ismember(truth,[0 1])) || any(~isfinite(score))
    error('GG:GroupMetric','Invalid label or score within a group.');
end
end
