function score = fuse_v43_probabilities(base_score, residual_score, alpha)
%FUSE_V43_PROBABILITIES Transparent convex probability fusion.

if ~isscalar(alpha) || ~isfinite(alpha) || alpha < 0 || alpha > 1
    error('Fusion alpha must be a finite scalar in [0,1].');
end
base_score = double(base_score(:));
residual_score = double(residual_score(:));
if numel(base_score) ~= numel(residual_score)
    error('Base and residual probability vectors must have equal length.');
end
if any(~isfinite(base_score)) || any(~isfinite(residual_score))
    error('Fusion inputs contain NaN or Inf.');
end
score = (1-alpha).*base_score + alpha.*residual_score;
score = min(max(score,0),1);
end
