 clc;
 clear all;
 close all;
testData = load('MNIST.mat'); 
nnOptions = {};
modelNN = learnNN(testData.X, testData.Y, nnOptions);
figure; 
plotConfMat(modelNN.confusion_valid);