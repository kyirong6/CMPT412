load lenet.mat
images = zeros(784,8);
for i = 1:8
   image_data = sprintf('../images/examples/%d.jpg', i);
   grey_image = rgb2gray(im2double(imread(image_data))).';
   image_stacked = reshape(grey_image, [], 1);
   images(:,i) = image_stacked; 
end
layers = get_lenet();
layers{1,1}.batch_size = size(images,2);
[output, P] = convnet_forward(params, layers, images);
disp(P)
[p, prediction] = max(P, [], 1);
disp(prediction-1)
disp(sum(prediction-1 == [1:8]));