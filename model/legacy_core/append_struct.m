function output = append_struct(output, row)
%APPEND_STRUCT Safely append a scalar structure.

if isempty(output)
    output = row;
    return;
end

existing_fields = sort(fieldnames(output));
new_fields = sort(fieldnames(row));

if ~isequal(existing_fields, new_fields)
    error('Cannot append structures with different fields.');
end

row = orderfields(row, output);
output(end+1, 1) = row;
end
