function metrics = binary_metrics(truth, score, threshold)
%BINARY_METRICS Binary metrics with average-precision AUPRC.

if nargin < 3 || isempty(threshold)
    threshold = 0.5;
end

truth = logical(truth(:));
score = double(score(:));

if numel(truth) ~= numel(score)
    error('truth and score must have the same length.');
end
if any(~isfinite(score))
    error('score contains NaN or Inf.');
end

prediction = score >= threshold;

TP = nnz(prediction & truth);
TN = nnz(~prediction & ~truth);
FP = nnz(prediction & ~truth);
FN = nnz(~prediction & truth);

safe_divide = @(a, b) a / max(b, eps);
precision = safe_divide(TP, TP + FP);
recall = safe_divide(TP, TP + FN);
specificity = safe_divide(TN, TN + FP);
accuracy = safe_divide(TP + TN, numel(truth));
balanced_accuracy = 0.5 * (recall + specificity);
F1 = safe_divide(2 * precision * recall, precision + recall);

mcc_denominator = sqrt(double(TP + FP) * double(TP + FN) ...
    * double(TN + FP) * double(TN + FN));
if mcc_denominator <= 0
    MCC = 0;
else
    MCC = (double(TP) * double(TN) - double(FP) * double(FN)) ...
        / mcc_denominator;
end

[sorted_score, order] = sort(score, 'descend');
sorted_truth = truth(order);
positive_count = nnz(sorted_truth);
negative_count = nnz(~sorted_truth);

tp_all = cumsum(sorted_truth);
fp_all = cumsum(~sorted_truth);
tie_ends = [sorted_score(1:end-1) ~= sorted_score(2:end); true];
idx = find(tie_ends);
tp_curve = double(tp_all(idx));
fp_curve = double(fp_all(idx));

if positive_count > 0 && negative_count > 0
    tpr = [0; tp_curve / positive_count];
    fpr = [0; fp_curve / negative_count];
    AUROC = trapz(fpr, tpr);
else
    AUROC = NaN;
end

if positive_count > 0
    recall_curve = tp_curve / positive_count;
    precision_curve = tp_curve ./ max(tp_curve + fp_curve, eps);
    % Same stepwise definition as average_precision_score.
    AUPRC = sum(diff([0; recall_curve]) .* precision_curve);
else
    AUPRC = NaN;
end

metrics = struct();
metrics.Threshold = threshold;
metrics.TP = TP;
metrics.TN = TN;
metrics.FP = FP;
metrics.FN = FN;
metrics.Accuracy = accuracy;
metrics.BalancedAccuracy = balanced_accuracy;
metrics.Precision = precision;
metrics.Recall = recall;
metrics.Specificity = specificity;
metrics.F1 = F1;
metrics.MCC = MCC;
metrics.AUROC = AUROC;
metrics.AUPRC = AUPRC;
metrics.Brier = mean((score - double(truth)).^2);
metrics.MAE = mean(abs(score - double(truth)));
end
