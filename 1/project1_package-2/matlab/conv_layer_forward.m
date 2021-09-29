function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
feature_map_size = h_out * w_out;


assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in; 
input_n.width = w_in;  
input_n.channel = c;   

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 
output.data = zeros(feature_map_size * num, batch_size);
for sample_idx = 1:batch_size
    img = input.data(:, sample_idx);
    padded_img = padarray(reshape(img, [h_in,w_in,c]), [pad,pad], 0 ,'both'); %transpose
    
    if c > 1
        padded_img = permute(padded_img, [2,1,3]);
    end
    if c == 1
        padded_img = padded_img';
    end

    side_edge = size(padded_img, 2);
    bottom_edge = size(padded_img, 1);
    tmp.data = zeros(feature_map_size, num);

    for filter_idx = 1:num % each filter
        weights = param.w(:, filter_idx);
        weights_grid = reshape(weights, [k,k,c]); %transpose
        if c > 1
            weights_grid  = permute(weights_grid , [2,1,3]);
        end
        if c == 1
            weights_grid  = weights_grid';
        end 
        row_top = 1;
        row_bottom = k;
        col_beg = 1;
        col_end = k;
        current_feature.data = zeros(feature_map_size, 1);
        for feature = 1: feature_map_size
            current_grid = padded_img(row_top:row_bottom, col_beg:col_end,:);
            feature_val =  sum(sum(dot(current_grid, weights_grid))) + param.b(filter_idx);
            current_feature.data(feature) = feature_val;

            if mod(col_end, side_edge) == 0 % right edge
                col_beg = 1;
                col_end = k;
                row_top = row_top + 1;
                row_bottom = row_bottom + 1;
                continue;
            end

            col_beg = col_beg + 1;
            col_end = col_end + 1;
        end
        tmp.data(:,filter_idx) = current_feature.data;
    end

    output.data(:,sample_idx) = reshape(tmp.data, [feature_map_size * num, 1]);
    output.height = h_out;
    output.width = w_out;
    output.channel = num;
end

