function [threshold, best_value] = select_threshold(truth, score, metric_name)
%SELECT_THRESHOLD Choose a classification threshold using validation only.

truth = logical(truth(:));
score = double(score(:));
metric_name = upper(metric_name);

[sorted_score, order] = sort(score, 'descend');
sorted_truth = truth(order);

tp = cumsum(sorted_truth);
fp = cumsum(~sorted_truth);
P = nnz(truth);
N = nnz(~truth);
fn = P - tp;
tn = N - fp;

last_of_tie = [sorted_score(1:end-1) ~= sorted_score(2:end); true];
idx = find(last_of_tie);

tp = double(tp(idx));
fp = double(fp(idx));
fn = double(fn(idx));
tn = double(tn(idx));
candidate_thresholds = sorted_score(idx);

switch metric_name
    case 'MCC'
        denominator = sqrt((tp + fp) .* (tp + fn) .* (tn + fp) .* (tn + fn));
        values = zeros(size(tp));
        valid = denominator > 0;
        values(valid) = (tp(valid).*tn(valid) - fp(valid).*fn(valid)) ...
            ./ denominator(valid);

    case 'F1'
        precision = tp ./ max(tp + fp, eps);
        recall = tp ./ max(tp + fn, eps);
        values = 2*precision.*recall ./ max(precision + recall, eps);

    case 'BALANCEDACCURACY'
        recall = tp ./ max(tp + fn, eps);
        specificity = tn ./ max(tn + fp, eps);
        values = 0.5*(recall + specificity);

    otherwise
        error('Unsupported threshold metric: %s', metric_name);
end

best_value = max(values);
ties = find(abs(values - best_value) <= 1e-12);
[~, closest] = min(abs(candidate_thresholds(ties) - 0.5));
selected = ties(closest);
threshold = candidate_thresholds(selected);

if isempty(threshold) || ~isfinite(threshold)
    threshold = 0.5;
    best_value = NaN;
end
end
