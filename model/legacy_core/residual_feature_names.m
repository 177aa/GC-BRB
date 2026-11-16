function names = residual_feature_names(data)
%RESIDUAL_FEATURE_NAMES Names matching evidence.feature_matrix columns.

layers = string(data.layer_names(:));
names = "OmicsScore";

prefixes = ["RWR_"; "OneHop_"; "TwoHop_"; ...
    "Confidence_"; "Presence_"];
for p = 1:numel(prefixes)
    names = [names; prefixes(p) + layers]; %#ok<AGROW>
end
names = [names; "WeightedRWR"; "WeightedOneHop"; ...
    "WeightedTwoHop"; "NetworkCoverage"; ...
    "RWRConflict"; "HopConflict"];
names = names(:);
end
