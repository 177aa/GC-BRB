function q = empirical_quantile(values, probability)
%EMPIRICAL_QUANTILE Toolbox-free linearly interpolated sample quantile.

values = sort(double(values(:)));
values = values(isfinite(values));

if isempty(values)
    q = NaN;
    return;
end

probability = min(max(double(probability), 0), 1);
if numel(values) == 1
    q = values(1);
    return;
end

position = 1 + (numel(values) - 1) * probability;
lower_index = floor(position);
upper_index = ceil(position);

if lower_index == upper_index
    q = values(lower_index);
else
    fraction = position - lower_index;
    q = values(lower_index) * (1 - fraction) + values(upper_index) * fraction;
end
end
