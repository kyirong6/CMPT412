%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix

predictions = zeros(1, size(xtest,2));
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [p, prediction] = max(P, [], 1);
    predictions(:, i:i+99) = prediction;
end

%disp(predictions)
%disp(ytest)
C = confusionmat(ytest, predictions);
confusionchart(C, [0:9])