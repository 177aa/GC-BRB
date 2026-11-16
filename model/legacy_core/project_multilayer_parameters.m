function x = project_multilayer_parameters(x, model)
%PROJECT_MULTILAYER_PARAMETERS Box and rule-belief constraints.

x = min(max(double(x(:)),0),1);
if numel(x) ~= model.total_dim
    error('Parameter dimension mismatch.');
end

for l = 1:model.num_layers
    start = model.layer(l).start;
    beta_count = model.num_rules*model.num_outputs;
    beta = reshape(x(start:start+beta_count-1), ...
        model.num_outputs, model.num_rules).';
    for r = 1:model.num_rules
        total = sum(beta(r,:));
        if total <= model.eps
            beta(r,:) = 1/model.num_outputs;
        else
            beta(r,:) = beta(r,:)/total;
        end
    end
    x(start:start+beta_count-1) = reshape(beta.',[],1);
end
end
