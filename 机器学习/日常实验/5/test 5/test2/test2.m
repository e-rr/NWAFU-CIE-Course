clc
close all; 
clear all;
load('MNIST.mat');
inputs = X;
targets = Y;
goal = 0;
spread = 2.750;
MN = 10;                                                 
DF = 2;   
epochs = 100;
acc = 0;
for i = 1:epochs
    [m, n] = size(inputs);
    [o, p] = size(targets);
    P = 0.70;
    idx = randperm(m);
    trainInputs = inputs(idx(1:round(P*m)),:)'; 
    trainInputs = mapminmax(trainInputs);
    trainTargets = targets(idx(1:round(P*o)),:)';
    testInputs = inputs(idx(round(P*m)+1:end),:)';
    testInputs = mapminmax(testInputs);
    testTargets = targets(idx(round(P*o)+1:end),:)';
    net = newrb(trainInputs,trainTargets,goal,spread,MN,DF);
    testOutputs = net(testInputs);
    e = gsubtract(testTargets,testOutputs);
    performance = perform(net,testTargets,testOutputs);
    tind = vec2ind(testTargets);
    yind = vec2ind(testOutputs);
    prs = sum(tind ~= yind)/numel(tind);
    acc = 100 * (1 - prs);
    fprintf('Acc = %.3f%%\n', acc);
end
save('net.mat','net');  
acc = acc/epochs;
fprintf('平均Acc = %.3f%%\n', acc);