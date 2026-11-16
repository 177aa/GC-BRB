function [score, details] = brb_layer_forward(X, references, rule_antecedents, parameters, eps_value)
%BRB_LAYER_FORWARD Forward inference for one two-input, nine-rule BRB layer.

if nargin < 5
    eps_value = 1e-12;
end
if size(X, 2) ~= 2
    error('Each hierarchical BRB layer requires exactly two inputs.');
end

num_samples = size(X, 1);
num_rules = size(rule_antecedents, 1);
num_outputs = size(parameters.beta, 2);

membership_1 = linear_membership(X(:, 1), references(:, 1));
membership_2 = linear_membership(X(:, 2), references(:, 2));

attribute_weights = max(parameters.attribute_weights(:), 0);
max_weight = max(attribute_weights);
if max_weight <= eps_value
    normalized_attribute_weights = ones(2, 1);
else
    normalized_attribute_weights = attribute_weights / max_weight;
end

matching = ones(num_samples, num_rules);
for k = 1:num_rules
    mu1 = membership_1(:, rule_antecedents(k, 1));
    mu2 = membership_2(:, rule_antecedents(k, 2));

    if normalized_attribute_weights(1) <= eps_value
        term1 = ones(num_samples, 1);
    else
        term1 = mu1 .^ normalized_attribute_weights(1);
    end

    if normalized_attribute_weights(2) <= eps_value
        term2 = ones(num_samples, 1);
    else
        term2 = mu2 .^ normalized_attribute_weights(2);
    end

    matching(:, k) = term1 .* term2;
end

rule_weights = max(parameters.rule_weights(:).', eps_value);
unnormalized_activation = matching .* rule_weights;
activation_sum = sum(unnormalized_activation, 2);

activation = zeros(size(unnormalized_activation));
valid_rows = activation_sum > eps_value & isfinite(activation_sum);
activation(valid_rows, :) = unnormalized_activation(valid_rows, :) ...
    ./ activation_sum(valid_rows);

if any(~valid_rows)
    activation(~valid_rows, :) = 1 / num_rules;
end

% Analytical evidential-reasoning aggregation for complete rule beliefs.
temp3 = prod(1 - activation, 2);
temp1 = zeros(num_samples, num_outputs);

for j = 1:num_outputs
    belief_term = activation .* parameters.beta(:, j).' + 1 - activation;
    temp1(:, j) = prod(belief_term, 2);
end

normalizer_denominator = sum(temp1, 2) - (num_outputs - 1) * temp3;
normalizer_denominator = max(normalizer_denominator, eps_value);
normalizer = 1 ./ normalizer_denominator;

output_denominator = 1 - normalizer .* temp3;
output_denominator = max(output_denominator, eps_value);

belief = (normalizer .* (temp1 - temp3)) ./ output_denominator;
belief = max(belief, 0);
belief_sum = sum(belief, 2);

invalid_belief = ~isfinite(belief_sum) | belief_sum <= eps_value;
if any(invalid_belief)
    fallback_positive = sum(activation(invalid_belief, :) ...
        .* parameters.beta(:, 2).', 2);
    belief(invalid_belief, 2) = min(max(fallback_positive, 0), 1);
    belief(invalid_belief, 1) = 1 - belief(invalid_belief, 2);
    belief_sum(invalid_belief) = 1;
end

belief = belief ./ belief_sum;
score = belief(:, 2);

details = struct();
details.membership_1 = membership_1;
details.membership_2 = membership_2;
details.matching = matching;
details.activation = activation;
details.belief = belief;
end
