function [model, score_all, coefficient_table] = fit_omics_evidence( ...
    X, y, train_idx, cfg, lambda)
%FIT_OMICS_EVIDENCE Fit a transparent class-balanced ridge logistic model.

if nargin < 5 || isempty(lambda)
    if isfield(cfg, 'omics_lambda')
        lambda = cfg.omics_lambda;
    else
        lambda = 1e-4;
    end
end

Xtrain = double(X(train_idx, :));
ytrain = double(y(train_idx));

if cfg.omics_standardize
    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1);
    sigma(sigma < 1e-12) = 1;
    Xtrain_fit = (Xtrain - mu) ./ sigma;
    Xall_fit = (double(X) - mu) ./ sigma;
else
    mu = zeros(1, size(X,2));
    sigma = ones(1, size(X,2));
    Xtrain_fit = Xtrain;
    Xall_fit = double(X);
end

num_pos = nnz(ytrain == 1);
num_neg = nnz(ytrain == 0);
if num_pos == 0 || num_neg == 0
    error('Omics fitting requires both classes.');
end
weights = zeros(size(ytrain));
weights(ytrain == 1) = 0.5 / num_pos;
weights(ytrain == 0) = 0.5 / num_neg;
weights = weights * numel(weights);

classifier = fitclinear(Xtrain_fit, ytrain, ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge', ...
    'Lambda', lambda, ...
    'Solver', 'lbfgs', ...
    'Weights', weights, ...
    'ClassNames', [0; 1], ...
    'Verbose', 0);

[~, raw_scores] = predict(classifier, Xall_fit);
pos_col = find(double(classifier.ClassNames) == 1, 1);
if isempty(pos_col)
    error('Cannot identify positive logistic class column.');
end
score_all = double(raw_scores(:, pos_col));
if ~all(score_all >= 0 & score_all <= 1)
    score_all = 1 ./ (1 + exp(-min(max(score_all, -40), 40)));
end
score_all = min(max(score_all, 0), 1);

model = struct();
model.classifier = classifier;
model.mu = mu;
model.sigma = sigma;
model.lambda = lambda;

beta_standardized = classifier.Beta(:);
beta_original = beta_standardized ./ sigma(:);
bias_original = classifier.Bias ...
    - sum((mu(:) ./ sigma(:)) .* beta_standardized);
coefficient_table = table((1:numel(beta_original)).', ...
    beta_standardized, beta_original, ...
    'VariableNames', {'FeatureIndex','StandardizedCoefficient', ...
    'OriginalScaleCoefficient'});
coefficient_table.BiasOriginalScale = repmat( ...
    bias_original, height(coefficient_table), 1);
coefficient_table.Lambda = repmat(lambda, ...
    height(coefficient_table), 1);
end
