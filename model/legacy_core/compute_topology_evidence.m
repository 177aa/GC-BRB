function topology = compute_topology_evidence(data, seed_idx, cfg, print_prefix)
%COMPUTE_TOPOLOGY_EVIDENCE Build one-hop, two-hop and RWR evidence.
%
% Only labels in seed_idx are used. Validation and test labels never enter
% any topology calculation. Exact duplicate local nodes share restart mass.

if nargin < 4
    print_prefix = '';
end

num_global = data.num_global_genes;
num_layers = data.num_layers;
seed_flag = false(num_global, 1);
seed_flag(seed_idx) = true;

onehop = 0.5 * ones(num_global, num_layers);
twohop = 0.5 * ones(num_global, num_layers);
diffusion = 0.5 * ones(num_global, num_layers);
confidence = zeros(num_global, num_layers);
rwr_iterations = zeros(num_layers, 1);

for s = 1:num_layers
    layer = data.layers{s};
    g = double(layer.local_to_global(:));
    local_seed = seed_flag(g);
    local_y = data.y(g);

    global_local_count = accumarray(g, 1, [num_global, 1], @sum, 0);
    local_gene_weight = 1 ./ max(global_local_count(g), 1);

    positive_seed = double(local_seed & local_y == 1) .* local_gene_weight;
    negative_seed = double(local_seed & local_y == 0) .* local_gene_weight;

    if sum(positive_seed) <= 0 || sum(negative_seed) <= 0
        warning('%s lacks one seed class; topology evidence is neutral.', ...
            layer.name);
        continue;
    end

    [rwr_score, rwr_mass, iterations] = random_walk_scores( ...
        layer.A, layer.degree, positive_seed, negative_seed, cfg);
    rwr_iterations(s) = iterations;

    pos_norm = positive_seed / sum(positive_seed);
    neg_norm = negative_seed / sum(negative_seed);
    source_degree = max(layer.degree(:), 1);

    pos_h1 = layer.A * (pos_norm ./ source_degree);
    neg_h1 = layer.A * (neg_norm ./ source_degree);
    h1_sum = pos_h1 + neg_h1;
    h1_score = 0.5 * ones(layer.num_genes, 1);
    valid_h1 = h1_sum > cfg.neighbor_smoothing;
    h1_score(valid_h1) = pos_h1(valid_h1) ./ h1_sum(valid_h1);

    pos_h2 = layer.A * (pos_h1 ./ source_degree);
    neg_h2 = layer.A * (neg_h1 ./ source_degree);
    h2_sum = pos_h2 + neg_h2;
    h2_score = 0.5 * ones(layer.num_genes, 1);
    valid_h2 = h2_sum > cfg.neighbor_smoothing;
    h2_score(valid_h2) = pos_h2(valid_h2) ./ h2_sum(valid_h2);

    degree_q90 = empirical_quantile(layer.degree, 0.90);
    if degree_q90 <= 0
        degree_conf = zeros(layer.num_genes, 1);
    else
        degree_conf = min(1, log1p(layer.degree) / log1p(degree_q90));
    end

    positive_mass = rwr_mass(rwr_mass > 0);
    if isempty(positive_mass)
        mass_conf = zeros(layer.num_genes, 1);
    else
        mass_q90 = empirical_quantile(positive_mass, 0.90);
        mass_conf = min(1, rwr_mass / max(mass_q90, cfg.eps));
    end

    local_support = h1_sum + h2_sum;
    support_positive = local_support(local_support > 0);
    if isempty(support_positive)
        support_conf = zeros(layer.num_genes, 1);
    else
        support_q90 = empirical_quantile(support_positive, 0.90);
        support_conf = min(1, local_support ...
            / max(support_q90, cfg.eps));
    end

    local_confidence = 0.10 + 0.20 * degree_conf ...
        + 0.35 * mass_conf + 0.35 * support_conf;
    local_confidence = min(max(local_confidence, 0), 1);

    [unique_g, ~, group] = unique(g, 'stable');
    pooled_h1 = accumarray(group, h1_score, [], @mean);
    pooled_h2 = accumarray(group, h2_score, [], @mean);
    pooled_rwr = accumarray(group, rwr_score, [], @mean);
    pooled_confidence = accumarray(group, local_confidence, [], @mean);

    onehop(unique_g, s) = pooled_h1;
    twohop(unique_g, s) = pooled_h2;
    diffusion(unique_g, s) = pooled_rwr;
    confidence(unique_g, s) = pooled_confidence;

    if cfg.verbose && (strncmp(print_prefix, 'Full', 4) || cfg.crossfit_verbose)
        unique_positive = unique(g(positive_seed > 0));
        unique_negative = unique(g(negative_seed > 0));
        fprintf(['  %sEvidence %-10s seeds +%d/-%d, RWR=%d, ' ...
            'mean confidence=%.3f\n'], ...
            print_prefix, layer.name, numel(unique_positive), ...
            numel(unique_negative), iterations, mean(pooled_confidence));
    end
end

confidence(~data.presence) = 0;

topology = struct();
topology.onehop = onehop;
topology.twohop = twohop;
topology.diffusion = diffusion;
topology.confidence = confidence;
topology.rwr_iterations = rwr_iterations;
end
