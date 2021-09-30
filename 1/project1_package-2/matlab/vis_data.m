layers = get_lenet();
load lenet.mat

global resultsdir
resultsdir = '../results';
[~,~,~] = mkdir(resultsdir);

% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
figure();
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
% Fill in your code here to plot the features.
%disp(output_1)
%size(output_1)
%disp(output{3})

conv_outputs = output{2}.data;
relu_outputs = output{3}.data;

% conv images
conv_outputs_reshaped = reshape(conv_outputs, [24,24,20]);
current_img = 1;
fig = figure;
for i = 1:4
    for j = 1:5
        subplot(4,5, current_img);
        imshow(conv_outputs_reshaped(:,:,current_img).');
        current_img = current_img + 1;
    end
end
filename = [resultsdir sprintf('/%s.png', 'conv_outputs')];
frame = getframe(fig);
imwrite(frame2im(frame), filename);


% relu images

relu_outputs_reshaped = reshape(relu_outputs, [24,24,20]);
relu_outputs(relu_outputs>0) = 0;
relu_outputs(relu_outputs<0) = 1;
current_img = 1;
fig = figure;
for i = 1:4
    for j = 1:5
        subplot(4,5, current_img);
        imshow(relu_outputs_reshaped(:,:,current_img).');
        current_img = current_img + 1;
    end
end
filename = [resultsdir sprintf('/%s.png', 'relu_outputs')];
frame = getframe(fig);
imwrite(frame2im(frame), filename);






