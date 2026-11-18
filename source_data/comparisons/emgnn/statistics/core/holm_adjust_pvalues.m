function adjusted = holm_adjust_pvalues(p)
%HOLM_ADJUST_PVALUES Holm step-down family-wise error correction.

p = double(p(:));
if any(~isfinite(p)) || any(p<0) || any(p>1)
    error('p values must be finite and between 0 and 1.');
end

m = numel(p);
[sorted_p,order] = sort(p,'ascend');

adjusted_sorted = zeros(m,1);
running_max = 0;

for i = 1:m
    candidate = (m-i+1)*sorted_p(i);
    running_max = max(running_max,candidate);
    adjusted_sorted(i) = min(1,running_max);
end

adjusted = zeros(m,1);
adjusted(order) = adjusted_sorted;
end
