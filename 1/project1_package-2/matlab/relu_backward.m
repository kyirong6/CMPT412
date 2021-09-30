function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
relu = max(input.data, 0);
applied = relu == input.data; 
input_od = output.diff .* applied;
end
