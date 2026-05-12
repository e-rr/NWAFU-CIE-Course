clc;
clear all;
close all;
load('MNIST.mat')
dat = X(1:3000,:)';
dalabels=Y(1:3000,:);
dat=dat'>0;     
trat=dat(1:2800,:);
tabels=dalabels(1:2800,:);
croset=dat(2001:3000,:);
crolabels=dalabels(2001:3000,:);
[sample_size, fze]=size(trat);
classes=10;
priors = zeros(classes, 1);   
eve = zeros(fze, 1);                                        
ls = zeros(classes, fze);
 accuracy = 0.0;
    k = 0.0;
    k_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 60];
    for i=1:length(k_values)
        [c_likelihoods, c_priors, c_evidence] = mynaive_bayest(trat, tabels, classes, k_values(i));
        [crossval_predicted_classes, crossval_posteriors] = mynaive_bayesc(croset, c_priors, c_likelihoods, c_evidence);
        c_accuracy = sum(crossval_predicted_classes == crolabels)/length(crolabels)*100.0;
        if c_accuracy>accuracy
            accuracy = c_accuracy;
            k = k_values(i);
            ls = c_likelihoods;
            priors = c_priors;
            eve = c_evidence;
        end
    end
    [tpredictedclasses, ts] = mynaive_bayesc(trat, priors, ls, eve);
    accuracy = sum(tpredictedclasses == tabels)/length(tabels)*100.0;
    fprintf("准确度=%2.2f%%\n", accuracy);

A=categorical(tpredictedclasses);
B=categorical(tabels);
plotconfusion(A,B)
