function [fused_score, details] = er_fuse_network_beliefs( ...
    network_score, confidence, presence, global_reliability, eps_value)
%ER_FUSE_NETWORK_BELIEFS Reliability-discounted ER across PPI layers.

if nargin < 5
    eps_value = 1e-12;
end
[N,S] = size(network_score);
if ~isequal(size(confidence),[N,S]) || ~isequal(size(presence),[N,S])
    error('Network evidence arrays must have equal size.');
end

global_reliability = min(max(global_reliability(:).',0),1);
if numel(global_reliability) ~= S
    error('Network reliability length mismatch.');
end

weights = confidence .* double(presence) .* global_reliability;
weights = min(max(weights,0),1);

mass_non = zeros(N,1);
mass_can = zeros(N,1);
mass_unknown = ones(N,1);
max_conflict = zeros(N,1);

for s = 1:S
    w = weights(:,s);
    p = min(max(network_score(:,s),0),1);
    src_non = w .* (1-p);
    src_can = w .* p;
    src_unknown = 1-w;

    conflict = mass_non.*src_can + mass_can.*src_non;
    denominator = max(1-conflict, eps_value);

    new_non = (mass_non.*src_non + mass_non.*src_unknown ...
        + mass_unknown.*src_non) ./ denominator;
    new_can = (mass_can.*src_can + mass_can.*src_unknown ...
        + mass_unknown.*src_can) ./ denominator;
    new_unknown = (mass_unknown.*src_unknown) ./ denominator;

    mass_non = new_non;
    mass_can = new_can;
    mass_unknown = new_unknown;
    max_conflict = max(max_conflict, conflict);
end

% Retain unresolved belief as epistemic uncertainty. Assigning the
% unknown mass to the midpoint makes low-reliability evidence shrink toward
% 0.5 instead of being artificially renormalized to an overconfident score.
fused_score = mass_can + 0.5*mass_unknown;
fused_score = min(max(fused_score,0),1);

details = struct();
details.weights = weights;
details.mass_non_cancer = mass_non;
details.mass_cancer = mass_can;
details.mass_unknown = mass_unknown;
details.max_conflict = max_conflict;
details.coverage = mean(presence,2);
end
