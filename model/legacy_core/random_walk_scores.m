function [score, mass, iterations] = random_walk_scores( ...
    A, degree, positive_seed, negative_seed, cfg)
%RANDOM_WALK_SCORES Class-balanced random walk with restart.
%
% A contains no self loops. The transition includes one self loop per node.
% Positive and negative restart vectors are normalized separately, so the
% large negative class does not dominate merely through class frequency.

n = size(A, 1);
positive_seed = double(positive_seed(:));
negative_seed = double(negative_seed(:));

if numel(positive_seed) ~= n || numel(negative_seed) ~= n
    error('Seed vectors must match adjacency size.');
end

if sum(positive_seed) > 0
    positive_seed = positive_seed / sum(positive_seed);
end
if sum(negative_seed) > 0
    negative_seed = negative_seed / sum(negative_seed);
end

pos = positive_seed;
neg = negative_seed;
restart = cfg.rwr_restart;
degree_self = degree(:) + 1;

for iteration = 1:cfg.rwr_max_iterations
    pos_next = restart * positive_seed ...
        + (1 - restart) * (A * (pos ./ degree_self) + pos ./ degree_self);
    neg_next = restart * negative_seed ...
        + (1 - restart) * (A * (neg ./ degree_self) + neg ./ degree_self);

    delta = max(norm(pos_next - pos, 1), norm(neg_next - neg, 1));
    pos = pos_next;
    neg = neg_next;
    if delta < cfg.rwr_tolerance
        break;
    end
end

mass = pos + neg;
score = 0.5 * ones(n, 1);
valid = mass > cfg.eps;
score(valid) = pos(valid) ./ mass(valid);
score = min(max(score, 0), 1);
iterations = iteration;
end
