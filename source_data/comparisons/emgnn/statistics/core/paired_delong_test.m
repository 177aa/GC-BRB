function result = paired_delong_test(truth,score1,score2)
%PAIRED_DELONG_TEST Two-sided DeLong test for two correlated AUROCs.
%
% truth  : binary labels
% score1 : model 1 scores
% score2 : model 2 scores
%
% The implementation uses the positive/negative placement values:
%   V10_i = mean_j phi(X_i,Y_j)
%   V01_j = mean_i phi(X_i,Y_j)
% and the DeLong covariance estimate.

truth = logical(truth(:));
score1 = double(score1(:));
score2 = double(score2(:));

if numel(truth) ~= numel(score1) || numel(truth) ~= numel(score2)
    error('truth, score1 and score2 must have equal length.');
end
if any(~isfinite(score1)) || any(~isfinite(score2))
    error('DeLong input contains NaN or Inf.');
end

positive = [score1(truth),score2(truth)];
negative = [score1(~truth),score2(~truth)];

m = size(positive,1);
n = size(negative,1);

if m < 2 || n < 2
    error('Paired DeLong requires at least two positives and negatives.');
end

v10 = zeros(m,2);
v01 = zeros(n,2);
auc = zeros(2,1);

for k = 1:2
    difference = positive(:,k)-negative(:,k).';
    phi = double(difference>0) + 0.5*double(difference==0);

    v10(:,k) = mean(phi,2);
    v01(:,k) = mean(phi,1).';
    auc(k) = mean(v10(:,k));
end

s10 = cov(v10,0);
s01 = cov(v01,0);
sigma = s10/m + s01/n;

contrast = [1;-1];
variance = contrast.'*sigma*contrast;
delta = auc(1)-auc(2);

if variance <= eps
    if abs(delta) <= eps
        z = 0;
        p = 1;
    else
        z = sign(delta)*Inf;
        p = 0;
    end
else
    z = delta/sqrt(variance);
    p = erfc(abs(z)/sqrt(2));
end

result = struct();
result.AUC1 = auc(1);
result.AUC2 = auc(2);
result.Delta = delta;
result.Variance = variance;
result.StandardError = sqrt(max(variance,0));
result.Z = z;
result.PValue = min(max(p,0),1);
result.Covariance = sigma;
end
