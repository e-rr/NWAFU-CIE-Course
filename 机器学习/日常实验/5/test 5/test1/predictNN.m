function [p, h] = predictNN(X, varargin)
if nargin==2
    modelNN = varargin{1};
    Theta = modelNN.Theta;
    activationFn = modelNN.activationFn;
    X = (X - repmat(modelNN.nnMu,size(X,1),1))...
        ./repmat(modelNN.nnSigma,size(X,1),1);
    X(isnan(X)) = 0;
elseif nargin==3
    Theta = varargin{1};
    activationFn = varargin{2};
end
if strcmp(activationFn, 'tanh')
    aF = @tanh;
else 
    aF = @sigmoid;
end
m = size(X, 1);
h = aF([ones(m, 1) X] * Theta{1}');
for ii=2:numel(Theta)
    h = aF([ones(m, 1) h] * Theta{ii}');
end
[~, p] = max(h, [], 2);
end
