function membership = linear_membership(values, references)
%LINEAR_MEMBERSHIP Convert scalar evidence into Low/Medium/High beliefs.
%
% values     - S-by-1 vector
% references - 3-by-1 strictly increasing anchors
% membership - S-by-3 matrix whose rows sum to one

values = double(values(:));
references = double(references(:));
if numel(references) ~= 3
    error('Exactly three reference values are required.');
end
if ~(references(1) < references(2) && references(2) < references(3))
    error('Reference values must be strictly increasing: %s', ...
        mat2str(references.'));
end

membership = zeros(numel(values), 3);

low_mask = values <= references(1);
high_mask = values >= references(3);
middle_left = values > references(1) & values <= references(2);
middle_right = values > references(2) & values < references(3);

membership(low_mask, 1) = 1;
membership(high_mask, 3) = 1;

if any(middle_left)
    t = (values(middle_left) - references(1)) ...
        / (references(2) - references(1));
    membership(middle_left, 1) = 1 - t;
    membership(middle_left, 2) = t;
end

if any(middle_right)
    t = (values(middle_right) - references(2)) ...
        / (references(3) - references(2));
    membership(middle_right, 2) = 1 - t;
    membership(middle_right, 3) = t;
end

row_sums = sum(membership, 2);
invalid = row_sums <= 0 | ~isfinite(row_sums);
if any(invalid)
    membership(invalid, :) = 1/3;
    row_sums(invalid) = 1;
end
membership = membership ./ row_sums;
end
