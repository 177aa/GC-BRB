function loss = multilayer_training_loss_v3layer( ...
    x,evidence,data,model,idx,cfg)
%MULTILAYER_TRAINING_LOSS_V3LAYER Same formal V4 loss, new forward model.

score = multilayer_brb_forward_v3layer( ...
    x,evidence,model,idx);
y = double(data.y(idx));
score = min(max(score,1e-7),1-1e-7);

pos = y==1;
neg = y==0;
if ~any(pos) || ~any(neg)
    error('Training loss requires both classes.');
end

balanced_bce = -0.5*mean(log(score(pos))) ...
    -0.5*mean(log(1-score(neg)));

switch lower(cfg.loss)
    case 'bce'
        loss = -mean(y.*log(score) ...
            +(1-y).*log(1-score));

    case 'balanced_bce'
        loss = balanced_bce;

    case 'balanced_bce_rank'
        pair_loss = deterministic_pairwise_logistic_local( ...
            score(pos),score(neg), ...
            cfg.ranking_pairs_per_positive, ...
            cfg.ranking_temperature);
        loss = balanced_bce ...
            + cfg.ranking_loss_weight*pair_loss;

    otherwise
        error('Unsupported loss: %s',cfg.loss);
end

if ~isfinite(loss)
    loss = realmax('double')/100;
end
end

function value = deterministic_pairwise_logistic_local( ...
    positive_score,negative_score,pairs_per_positive,temperature)
positive_score = positive_score(:);
negative_score = negative_score(:);
pairs_per_positive = max(1,round(pairs_per_positive));
temperature = max(temperature,1e-6);

positive_repeated = repelem( ...
    positive_score,pairs_per_positive);
base = (0:numel(positive_repeated)-1).';
negative_index = mod(base*104729+37, ...
    numel(negative_score))+1;
margin = (positive_repeated ...
    - negative_score(negative_index))/temperature;
value = mean(max(-margin,0) ...
    + log1p(exp(-abs(margin))));
end
