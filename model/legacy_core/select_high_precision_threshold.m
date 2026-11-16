function result = select_high_precision_threshold( ...
    truth,score,target_precision,min_predicted_positive)
%SELECT_HIGH_PRECISION_THRESHOLD Validation-only high-precision threshold.
%
% Chooses the threshold giving the largest recall subject to:
%   precision >= target_precision
%   number of predicted positives >= min_predicted_positive
%
% If no threshold satisfies both constraints, Valid=false and Threshold=Inf.
% The target precision is NEVER relaxed automatically.

truth = double(truth(:));
score = double(score(:));

if numel(truth)~=numel(score)
    error('truth and score length mismatch.');
end
if any(~ismember(truth,[0 1])) || any(~isfinite(score))
    error('Invalid validation truth or score.');
end

candidate_thresholds = unique(score,'sorted');
candidate_thresholds = flipud(candidate_thresholds);

rows = struct([]);
for i = 1:numel(candidate_thresholds)
    t = candidate_thresholds(i);
    pred = score>=t;

    tp = nnz(pred & truth==1);
    fp = nnz(pred & truth==0);
    fn = nnz(~pred & truth==1);
    ncall = tp+fp;

    if ncall==0
        precision = NaN;
    else
        precision = tp/ncall;
    end
    recall = tp/max(tp+fn,1);

    row = struct();
    row.Threshold = t;
    row.PredictedPositive = ncall;
    row.TruePositive = tp;
    row.FalsePositive = fp;
    row.Precision = precision;
    row.Recall = recall;
    rows = append_local(rows,row);
end

T = struct2table(rows);
valid = T.PredictedPositive>=min_predicted_positive ...
    & T.Precision>=target_precision;

result = struct();
result.TargetPrecision = target_precision;
result.MinimumPredictedPositive = min_predicted_positive;
result.Valid = any(valid);
result.Curve = T;

if ~result.Valid
    result.Threshold = Inf;
    result.Precision = NaN;
    result.Recall = 0;
    result.PredictedPositive = 0;
    result.TruePositive = 0;
    result.FalsePositive = 0;
    return;
end

V = T(valid,:);
V.NegRecall = -V.Recall;
V.NegPrecision = -V.Precision;
V.NegThreshold = -V.Threshold;

% Highest recall first. If exact recall ties, prefer higher precision,
% then the more conservative (higher) threshold.
V = sortrows(V, ...
    {'NegRecall','NegPrecision','NegThreshold'}, ...
    {'ascend','ascend','ascend'});
best = V(1,:);

result.Threshold = best.Threshold;
result.Precision = best.Precision;
result.Recall = best.Recall;
result.PredictedPositive = best.PredictedPositive;
result.TruePositive = best.TruePositive;
result.FalsePositive = best.FalsePositive;
end

function output = append_local(output,row)
if isempty(output)
    output = row;
else
    row = orderfields(row,output);
    output(end+1,1) = row;
end
end
