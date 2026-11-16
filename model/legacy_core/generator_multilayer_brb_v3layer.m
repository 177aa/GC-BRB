function [x0,model] = generator_multilayer_brb_v3layer( ...
    data,evidence,split,cfg)
%GENERATOR_MULTILAYER_BRB_V3LAYER Initialize the simplified model.
%
% BRB1: one-hop + two-hop -> local topology belief
% BRB2: RWR + local topology -> per-network topology belief
% ER  : reliability-aware fusion of all six network beliefs
% QA  : coverage/conflict/unknown-mass quality discount
% BRB3: multi-omics + quality-adjusted fused topology -> cancer belief

model.num_outputs = 2;
model.num_rules = 9;
model.num_layers = 3;
model.num_networks = data.num_layers;
model.eps = cfg.eps;
model.structure_name = 'THREE_LAYER_GC_BRB';
model.structure_version = cfg.structure_version;
model.level_names = {'Low','Medium','High'};
model.output_names = {'Non-cancer','Cancer'};
model.rule_antecedents = [
    1 1
    1 2
    1 3
    2 1
    2 2
    2 3
    3 1
    3 2
    3 3
];
model.references = [0 0; 0.5 0.5; 1 1];
model.layer_names = {
    'LocalMultiscaleTopologyBRB'
    'SharedNetworkBRB'
    'FinalDecisionBRB'
};
model.layer_attribute_names = {
    {'One-hop evidence','Two-hop evidence'}
    {'Random-walk evidence','Local multiscale topology'}
    {'Raw-64 omics evidence','Quality-adjusted ER topology'}
};
model.layer_dim = model.num_rules*model.num_outputs ...
    + model.num_rules + 2;

cursor = 1;
for l = 1:model.num_layers
    model.layer(l).start = cursor;
    model.layer(l).stop = cursor + model.layer_dim - 1;
    cursor = model.layer(l).stop + 1;
end
model.network_reliability_indices = ...
    cursor:(cursor+model.num_networks-1);
model.total_dim = model.network_reliability_indices(end);

rule_score = sum(model.rule_antecedents-1,2)/4;
beta = [1-rule_score,rule_score];
segment = [reshape(beta.',[],1); ...
    ones(model.num_rules,1);ones(2,1)];

x0 = zeros(model.total_dim,1);
for l = 1:model.num_layers
    x0(model.layer(l).start:model.layer(l).stop) = segment;
end
x0(model.network_reliability_indices) = ...
    initial_network_reliability_local( ...
    evidence,data,split,model.num_networks);
x0 = project_multilayer_parameters(x0,model);
end

function reliability = initial_network_reliability_local( ...
    evidence,data,split,num_networks)
% Training-only initialization, identical to the original V4 protocol.
raw_auprc = nan(num_networks,1);
for s = 1:num_networks
    raw_score = (evidence.diffusion(split.train_idx,s) ...
        + evidence.onehop(split.train_idx,s) ...
        + evidence.twohop(split.train_idx,s))/3;
    metrics = binary_metrics( ...
        data.y(split.train_idx),raw_score,0.5);
    raw_auprc(s) = metrics.AUPRC;
end

minimum = min(raw_auprc);
maximum = max(raw_auprc);
if ~all(isfinite(raw_auprc)) ...
        || maximum <= minimum + 1e-12
    reliability = 0.8*ones(num_networks,1);
else
    reliability = 0.4 + 0.6*(raw_auprc-minimum) ...
        /(maximum-minimum);
end
end
