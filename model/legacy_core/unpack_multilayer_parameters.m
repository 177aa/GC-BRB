function p = unpack_multilayer_parameters(x, model)
%UNPACK_MULTILAYER_PARAMETERS Decode projected model parameters.

x = project_multilayer_parameters(x, model);
p = struct();
for l = 1:model.num_layers
    cursor = model.layer(l).start;
    beta_count = model.num_rules*model.num_outputs;
    p.layer(l).beta = reshape(x(cursor:cursor+beta_count-1), ...
        model.num_outputs, model.num_rules).';
    cursor = cursor+beta_count;
    p.layer(l).rule_weights = x(cursor:cursor+model.num_rules-1);
    cursor = cursor+model.num_rules;
    p.layer(l).attribute_weights = x(cursor:cursor+1);
end
p.network_reliability = x(model.network_reliability_indices);
end
