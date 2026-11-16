function [probability,details] = multilayer_brb_forward_v3layer( ...
    x,evidence,model,indices)
%MULTILAYER_BRB_FORWARD_V3LAYER Three-layer white-box inference.
%
% Unlike the original V4, the target network is not reintroduced after ER.
% The reliability-aware ER consensus is quality-discounted and passed
% directly to the final multi-omics/topology BRB.

if nargin < 4 || isempty(indices)
    indices = (1:numel(evidence.omics_score)).';
else
    indices = indices(:);
end

p = unpack_multilayer_parameters(x,model);
N = numel(indices);
S = model.num_networks;

local_topology = zeros(N,S);
network_score = zeros(N,S);

for s = 1:S
    local_input = [evidence.onehop(indices,s), ...
        evidence.twohop(indices,s)];
    local_topology(:,s) = brb_layer_forward( ...
        local_input,model.references,model.rule_antecedents, ...
        p.layer(1),model.eps);

    network_input = [evidence.diffusion(indices,s), ...
        local_topology(:,s)];
    network_score(:,s) = brb_layer_forward( ...
        network_input,model.references,model.rule_antecedents, ...
        p.layer(2),model.eps);
end

[fused_topology,fusion_details] = er_fuse_network_beliefs( ...
    network_score,evidence.confidence(indices,:), ...
    evidence.presence(indices,:),p.network_reliability,model.eps);

coverage_factor = 0.35 + 0.65*fusion_details.coverage;
conflict_factor = 1 - 0.50*min(max( ...
    fusion_details.max_conflict,0),1);
unknown_factor = 1 - 0.50*min(max( ...
    fusion_details.mass_unknown,0),1);
quality = coverage_factor.*conflict_factor.*unknown_factor;
quality = min(max(quality,0),1);

quality_adjusted_topology = 0.5 ...
    + quality.*(fused_topology-0.5);
quality_adjusted_topology = min(max( ...
    quality_adjusted_topology,0),1);

final_input = [evidence.omics_score(indices), ...
    quality_adjusted_topology];
[probability,final_details] = brb_layer_forward( ...
    final_input,model.references,model.rule_antecedents, ...
    p.layer(3),model.eps);
probability = min(max(probability,0),1);

details = struct();
details.local_topology = local_topology;
details.network_score = network_score;
details.fused_topology = fused_topology;
details.evidence_quality = quality;
details.quality_adjusted_topology = quality_adjusted_topology;
details.fusion = fusion_details;
details.final = final_details;
details.network_reliability = p.network_reliability;
details.structure = 'THREE_LAYER_GC_BRB';
end
