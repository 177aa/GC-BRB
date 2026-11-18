function result = paired_stratified_bootstrap_auprc( ...
    truth,score1,score2,num_resamples,confidence_level,random_seed)
%PAIRED_STRATIFIED_BOOTSTRAP_AUPRC
%
% Paired, class-stratified bootstrap for the difference:
%   AUPRC(score1) - AUPRC(score2)
%
% Positive and negative genes are independently sampled with replacement,
% while both models use exactly the same sampled genes in each resample.

truth = logical(truth(:));
score1 = double(score1(:));
score2 = double(score2(:));

if numel(truth) ~= numel(score1) || numel(truth) ~= numel(score2)
    error('truth, score1 and score2 must have equal length.');
end
if any(~isfinite(score1)) || any(~isfinite(score2))
    error('Bootstrap input contains NaN or Inf.');
end
if num_resamples < 100
    error('num_resamples must be at least 100.');
end
if confidence_level <= 0 || confidence_level >= 1
    error('confidence_level must be between 0 and 1.');
end

positive_idx = find(truth);
negative_idx = find(~truth);

m = numel(positive_idx);
n = numel(negative_idx);

if m < 2 || n < 2
    error('Bootstrap requires at least two positives and negatives.');
end

observed = average_precision_local(truth,score1) ...
    - average_precision_local(truth,score2);

rng(random_seed,'twister');
delta = zeros(num_resamples,1);

progress_step = max(1000,floor(num_resamples/10));

for b = 1:num_resamples
    sampled_positive = positive_idx(randi(m,m,1));
    sampled_negative = negative_idx(randi(n,n,1));
    idx = [sampled_positive;sampled_negative];

    yb = truth(idx);
    delta(b) = average_precision_local(yb,score1(idx)) ...
        - average_precision_local(yb,score2(idx));

    if mod(b,progress_step)==0 || b==num_resamples
        fprintf('    Bootstrap %d/%d\n',b,num_resamples);
    end
end

alpha = 1-confidence_level;
ci = percentile_linear_local(delta, ...
    [100*alpha/2,100*(1-alpha/2)]);

% Conservative two-sided empirical p value with +1 correction.
p_lower = (nnz(delta<=0)+1)/(num_resamples+1);
p_upper = (nnz(delta>=0)+1)/(num_resamples+1);
p_value = min(1,2*min(p_lower,p_upper));

result = struct();
result.ObservedDelta = observed;
result.CILower = ci(1);
result.CIUpper = ci(2);
result.PValue = p_value;
result.NumResamples = num_resamples;
result.RandomSeed = random_seed;
end

function ap = average_precision_local(truth,score)
truth = logical(truth(:));
score = double(score(:));

[sorted_score,order] = sort(score,'descend');
sorted_truth = truth(order);

positive_count = nnz(sorted_truth);
if positive_count==0
    ap = NaN;
    return;
end

tp = cumsum(sorted_truth);
fp = cumsum(~sorted_truth);

tie_ends = [sorted_score(1:end-1)~=sorted_score(2:end);true];
idx = find(tie_ends);

tp_curve = double(tp(idx));
fp_curve = double(fp(idx));

recall_curve = tp_curve/positive_count;
precision_curve = tp_curve./max(tp_curve+fp_curve,eps);

ap = sum(diff([0;recall_curve]).*precision_curve);
end

function q = percentile_linear_local(x,percent)
x = sort(double(x(:)));
N = numel(x);
q = nan(size(percent));

for i = 1:numel(percent)
    p = min(max(percent(i),0),100)/100;
    position = 1+(N-1)*p;
    lower = floor(position);
    upper = ceil(position);

    if lower==upper
        q(i) = x(lower);
    else
        weight = position-lower;
        q(i) = (1-weight)*x(lower)+weight*x(upper);
    end
end
end
