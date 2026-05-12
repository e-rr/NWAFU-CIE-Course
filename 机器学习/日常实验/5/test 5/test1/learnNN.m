function modelNN = learnNN(X, Y, varargin)
lambda = 0;
maxIter = 500;
nrOfLabels = numel(unique(Y));
hiddenLayers = round(sqrt(nrOfLabels*size(X, 2)));
activationFn = 'sigm';
validPercent = 20;
doNormalize = 1;
nnMu = 0;
nnSigma = 1;
if nargin>2
    nnOptions = varargin{1};
    if ~mod(numel(nnOptions), 2)
        for ii=1:2:numel(nnOptions)
            if isnumeric(nnOptions{ii + 1})
                nOfn = numel(nnOptions{ii+1});
                if nOfn>1
                    eval(sprintf(['%s=[' repmat('%d '.', nOfn, 1)' '];'], nnOptions{ii}, nnOptions{ii+1}));
                else
                    eval(sprintf('%s=%d;', nnOptions{ii}, nnOptions{ii+1}));
                end
            else
                eval(sprintf('%s=''%s'';', nnOptions{ii}, nnOptions{ii+1}));
            end
            
        end
    else
        error('Number of Options should be Even.')
    end
end

input_layer_size  = size(X, 2);
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 

modelNN.activationFn = activationFn;
modelNN.lambda = lambda;
modelNN.maxIter = maxIter;
modelNN.layers = layers;
modelNN.doNormalize = doNormalize;
X(isnan(X))=0;
Y = Y - min(Y) + 1; 
m = size(X, 1);
validation_set_size = round(m*validPercent/100); % e.g. 10% for the validation
rand_indices = randperm(m);
X = X(rand_indices, :);
Y = Y(rand_indices);
X_valid = X(1:validation_set_size, :);
Y_valid = Y(1:validation_set_size);
X(1:validation_set_size,:) = [];
Y(1:validation_set_size) = [];
modelNN.X = X;
modelNN.Y = Y;
modelNN.X_valid = X_valid;
modelNN.Y_valid = Y_valid;

if doNormalize
    X_norm = zeros(size(X));
    X_norm(:,1) = ones(size(X,1),1);
    [X_norm(:,1:end), nnMu, nnSigma] = featureNormalize(X(:,1:end));
else
    X_norm = X;
end

initial_nn_params = randInitializeWeights(layers);
options = optimset('MaxIter', maxIter);

cf = @(p) nnCostFunction(p, layers, X_norm, Y, lambda, activationFn); 

startT = cputime;
[theta, modelNN.costArray] = fmincg(cf, initial_nn_params, options);
elapsed = cputime - startT;
fprintf('Required CPU Time: %f\n', elapsed);

modelNN.Theta = unpackTheta( theta, layers );
if doNormalize
    X_valid_norm = (X_valid - repmat(nnMu,size(X_valid,1),1))...
                         ./repmat(nnSigma,size(X_valid,1),1);
else
    X_valid_norm = X_valid;
end
X_valid_norm(isnan(X_valid_norm))=0;

p_valid = predictNN(X_valid_norm, modelNN.Theta, modelNN.activationFn);
p_train = predictNN(X_norm, modelNN.Theta, modelNN.activationFn);

confusion_train = zeros(nrOfLabels, nrOfLabels);
confusion_valid = zeros(nrOfLabels, nrOfLabels);


for lp=1:nrOfLabels
    for la=1:nrOfLabels
        confusion_valid(la, lp) = numel(find(p_valid==lp&Y_valid==la));
        confusion_train(la, lp) = numel(find(p_train==lp&Y==la));
    end
end

modelNN.confusion_valid = confusion_valid;
modelNN.confusion_train = confusion_train;
modelNN.nnMu = nnMu;
modelNN.nnSigma = nnSigma;
modelNN.trainingTimestamp = datestr(now,'yy_mm_dd_HH_MM_SS');

if exist('savePath', 'var')
    save(savePath, 'modelNN');
end
end
