clc;
clear all;
close all;
load('MNIST.mat')
opts.alpha = 1e-1;
opts.batchsize = 150;
opts.numepochs = 3;
opts.imageDim = 28;
opts.imageChannel = 1;
opts.numClasses = 10;
opts.lambda = 0.0001; 
opts.momentum = .95;
opts.mom = 0.5;
opts.momIncrease = 20;
images =X;
images = reshape(images,opts.imageDim,opts.imageDim,1,[]);
labels = Y;
labels(labels==0) = 10; 
testImages = X;
testImages = reshape(testImages,opts.imageDim,opts.imageDim,1,[]);
testLabels =Y;
testLabels(testLabels==0) = 10; 
cnn.layers = {
    struct('type', 'c', 'numFilters', 6, 'filterDim', 5) 
    struct('type', 'p', 'poolDim', 2) 
    struct('type', 'c', 'numFilters', 8, 'filterDim', 5) 
    struct('type', 'p', 'poolDim', 2) 
};
cnn = InitializeParameters(cnn,opts);
cnnTrain(cnn,images,labels,testImages,testLabels);