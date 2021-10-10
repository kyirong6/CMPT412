function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride; 
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out, w_out, c, batch_size]);
    filter_square = layer.k ^ 2;

    channel_grid_size = h_out * w_out;
    tmp.data = zeros(c * channel_grid_size, batch_size);
    
    for sample_idx = 1:input.batch_size
        img = input.data(:, sample_idx);
        filter_top_left = 1;
        filter_top_right = k;
        filter_bottom_left = filter_top_left + input.width;
        filter_bottom_right = filter_top_right + input.width;
        
        pooled = zeros(1, channel_grid_size);
        current_pool = 1;
        current_channel = 1;
        for i = 1:c * channel_grid_size
            % ----------------------middle--------------------------
            if mod(filter_top_right, input.width) ~= 0
               filter_grid = [img(filter_top_left), img(filter_top_right), img(filter_bottom_left), img(filter_bottom_right)];
               maximum = max(filter_grid);
               pooled(current_pool) = maximum;
               filter_top_left = filter_top_left + stride;
               filter_top_right = filter_top_right + stride;
               filter_bottom_left = filter_bottom_left + stride;
               filter_bottom_right = filter_bottom_right + stride;
               current_pool = current_pool + 1;
               continue
            end
            % ----------------------edge--------------------------
            if mod(filter_bottom_right, input.width) == 0 && mod(filter_bottom_right, size(input.data, 1) / c) ~= 0
                filter_grid = [img(filter_top_left), img(filter_top_right), img(filter_bottom_left), img(filter_bottom_right)];
                maximum = max(filter_grid);
                pooled(current_pool) = maximum;
                current_pool = current_pool + 1;
                filter_top_left = filter_bottom_right + 1;
                filter_top_right = filter_bottom_right + k;
                filter_bottom_left = filter_top_left + input.width;
                filter_bottom_right = filter_top_right + input.width;
                continue
            end 
            % ----------------------right corners--------------------------
            if mod(filter_bottom_right, size(input.data, 1) / c) == 0 % right corners
               filter_grid = [img(filter_top_left), img(filter_top_right), img(filter_bottom_left), img(filter_bottom_right)];
               maximum = max(filter_grid);
               pooled(current_pool) = maximum;
               output.data(:,:,current_channel,sample_idx) = reshape(pooled, [channel_grid_size ^ .5,channel_grid_size ^ .5]);
               pooled = zeros(1,channel_grid_size);
               current_pool = 1;
               current_channel = current_channel + 1;
               filter_top_left = filter_bottom_right + 1;
               filter_top_right = filter_bottom_right + k;
               filter_bottom_left = filter_top_left + input.width;
               filter_bottom_right = filter_top_right + input.width;

            end
        end     
    tmp.data(:,sample_idx) = reshape(output.data(:,:,:,sample_idx), [c * channel_grid_size, 1]);
    end      
    output.data = tmp.data;
end

