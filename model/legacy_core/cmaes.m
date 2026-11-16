function [bestx, output] = cmaes(fitness_function, x0, options, project_function, validation_function)
%CMAES Constrained CMA-ES with validation-based model selection.
%
% Inputs:
%   fitness_function    handle returning the training loss (minimize)
%   x0                  initial column vector
%   options             struct: max_generations, sigma0, population_size,
%                       patience, verbose
%   project_function    handle applying parameter constraints
%   validation_function optional handle returning a validation loss
%
% The algorithm tracks the best training solution and, when a validation
% function is supplied, returns the best validation solution.

if nargin < 4 || isempty(project_function)
    project_function = @(x) x;
end
if nargin < 5
    validation_function = [];
end

xmean = project_function(double(x0(:)));
dimension = numel(xmean);

if isfield(options, 'population_size') && ~isempty(options.population_size)
    lambda = round(options.population_size);
else
    lambda = 4 + floor(3 * log(dimension));
end
lambda = max(lambda, 6);
mu = floor(lambda / 2);

raw_weights = log(mu + 0.5) - log((1:mu).');
weights = raw_weights / sum(raw_weights);
mueff = sum(weights)^2 / sum(weights.^2);

cc = (4 + mueff / dimension) ...
    / (dimension + 4 + 2 * mueff / dimension);
cs = (mueff + 2) / (dimension + mueff + 5);
c1 = 2 / ((dimension + 1.3)^2 + mueff);
cmu = min(1 - c1, ...
    2 * (mueff - 2 + 1 / mueff) / ((dimension + 2)^2 + mueff));
damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dimension + 1)) - 1) + cs;

sigma = options.sigma0;
pc = zeros(dimension, 1);
ps = zeros(dimension, 1);
B = eye(dimension);
D = ones(dimension, 1);
C = eye(dimension);
invsqrtC = eye(dimension);
chiN = sqrt(dimension) * (1 - 1/(4*dimension) + 1/(21*dimension^2));

best_train_x = xmean;
best_train_loss = fitness_function(best_train_x);
evaluation_count = 1;

if isempty(validation_function)
    best_validation_x = [];
    best_validation_loss = Inf;
else
    best_validation_x = xmean;
    best_validation_loss = validation_function(xmean);
end

history = nan(options.max_generations, 5);
no_improvement = 0;
last_eigen_update = 0;

for generation = 1:options.max_generations
    arz = randn(dimension, lambda);
    arx = zeros(dimension, lambda);
    fitness = zeros(lambda, 1);

    for k = 1:lambda
        candidate = xmean + sigma * B * (D .* arz(:, k));
        candidate = project_function(candidate);
        arx(:, k) = candidate;

        value = fitness_function(candidate);
        if ~isfinite(value)
            value = realmax('double') / 100;
        end
        fitness(k) = value;
    end
    evaluation_count = evaluation_count + lambda;

    [fitness, order] = sort(fitness, 'ascend');
    xold = xmean;
    selected = arx(:, order(1:mu));
    xmean = project_function(selected * weights);

    if fitness(1) < best_train_loss
        best_train_loss = fitness(1);
        best_train_x = arx(:, order(1));
    end

    ymean = (xmean - xold) / max(sigma, eps);
    ps = (1 - cs) * ps ...
        + sqrt(cs * (2 - cs) * mueff) * invsqrtC * ymean;

    hsig_left = sum(ps.^2) ...
        / max(1 - (1 - cs)^(2 * generation), eps) / dimension;
    hsig = hsig_left < (2 + 4 / (dimension + 1));

    pc = (1 - cc) * pc ...
        + hsig * sqrt(cc * (2 - cc) * mueff) * ymean;

    artmp = (selected - xold) / max(sigma, eps);
    C = (1 - c1 - cmu) * C ...
        + c1 * (pc * pc.' + (1 - hsig) * cc * (2 - cc) * C) ...
        + cmu * artmp * diag(weights) * artmp.';

    sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));
    sigma = min(max(sigma, 1e-12), 2);

    if generation - last_eigen_update > max(1, floor(lambda / ((c1 + cmu) * dimension * 10)))
        last_eigen_update = generation;
        C = (C + C.') / 2;
        [B, eigenvalues] = eig(C);
        eigenvalues = max(real(diag(eigenvalues)), 1e-20);
        D = sqrt(eigenvalues);
        invsqrtC = B * diag(1 ./ D) * B.';
    end

    validation_loss = NaN;
    if ~isempty(validation_function)
        generation_candidates = {arx(:, order(1)), xmean};
        improved = false;
        for c = 1:numel(generation_candidates)
            candidate_validation = validation_function(generation_candidates{c});
            if isfinite(candidate_validation) ...
                    && candidate_validation < best_validation_loss - 1e-12
                best_validation_loss = candidate_validation;
                best_validation_x = generation_candidates{c};
                improved = true;
            end
        end

        if improved
            no_improvement = 0;
        else
            no_improvement = no_improvement + 1;
        end
        validation_loss = best_validation_loss;
    end

    condition_number = max(D) / max(min(D), 1e-20);
    history(generation, :) = [generation, best_train_loss, ...
        validation_loss, sigma, condition_number];

    if options.verbose
        if isempty(validation_function)
            fprintf('  CMA-ES gen %3d/%3d: train=%.6f sigma=%.4g cond=%.3g\n', ...
                generation, options.max_generations, best_train_loss, ...
                sigma, condition_number);
        else
            fprintf('  CMA-ES gen %3d/%3d: train=%.6f val=%.6f sigma=%.4g\n', ...
                generation, options.max_generations, best_train_loss, ...
                best_validation_loss, sigma);
        end
    end

    if ~isempty(validation_function) && no_improvement >= options.patience
        if options.verbose
            fprintf('  Early stopping after %d generations without validation improvement.\n', ...
                options.patience);
        end
        break;
    end

    if condition_number > 1e14 || sigma <= 1e-12
        if options.verbose
            fprintf('  CMA-ES numerical stopping criterion reached.\n');
        end
        break;
    end
end

last_generation = generation;
history = history(1:last_generation, :);

if isempty(validation_function)
    bestx = best_train_x;
else
    bestx = best_validation_x;
end
bestx = project_function(bestx);

output = struct();
output.generations = last_generation;
output.evaluations = evaluation_count;
output.best_train_loss = best_train_loss;
output.best_validation_loss = best_validation_loss;
output.history = history;
output.population_size = lambda;
end
