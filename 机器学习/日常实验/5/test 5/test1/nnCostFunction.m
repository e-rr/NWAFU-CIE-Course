function [J, grad] = nnCostFunction(nn_params, layers, X, y, lambda, activationFn)
Theta = unpackTheta(nn_params, layers);
m = size(X, 1);
nl = numel(layers) - 1;
num_labels = layers(end);
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));
snel = cumsum([0 nel]); 
if strcmp(activationFn, 'tanh')
    aF = @tanh;
    gF = @tanhGradient;
    y_nn = -ones(m, num_labels);    
else 
    aF = @sigmoid;
    gF = @sigmoidGradient;
    y_nn = zeros(m, num_labels);    
end
a_t = X; 
a = cell(nl, 1);
z = cell(nl + 1, 1);
for ii=1:nl
    a{ii} = [ones(m, 1) a_t]; 
    z{ii+1} = a{ii}*Theta{ii}';
    a_t = aF(z{ii+1});
end
h_nn = a_t;
idx = sub2ind([m num_labels],1:m,y');
y_nn(idx) = 1;
if strcmp(activationFn, 'tanh')
    J_matrix = -((y_nn+1).*log((h_nn + 1)/2)/2 + (1 - (y_nn+1)/2).*log(1 - (h_nn+1)/2));
else
    J_matrix = -(y_nn.*log(h_nn) + (1 - y_nn).*log(1 - h_nn));
end
J_matrix(isnan(J_matrix)) = 0;
J = sum(J_matrix(:));
J = J/m;
regularization_term = lambda * sum(cellfun(@(t) sum(sum(t(:,2:end).^2)), Theta)) / (2*m);
J = J + regularization_term;

delta = cell(nl + 1, 1);
Delta = cell(nl, 1);
ThetaGrad  =  cell(nl, 1);
grad = zeros(snel(end), 1);
delta{nl+1} = h_nn - y_nn;
Delta{nl} = delta{nl+1}' * a{nl}; 
ThetaGrad{nl} = Delta{nl}/m + [zeros(layers(nl+1), 1) Theta{nl}(:, 2:end)]*lambda/m;
grad((snel(nl)+1):snel(nl+1)) = ThetaGrad{nl}(:);
for ii = 2:nl 
    bi = nl - ii + 2; 
    delta{bi} = (delta{bi + 1} * Theta{bi}(:, 2:end)) .* gF(z{bi}); 
    Delta{bi-1} = delta{bi}' * a{bi-1}; 
    ThetaGrad{bi-1} = Delta{bi-1}/m + [zeros(layers(bi), 1) Theta{bi-1}(:, 2:end)]*lambda/m; 
    grad((snel(bi - 1)+1):snel(bi)) = ThetaGrad{bi-1}(:);
end
end
