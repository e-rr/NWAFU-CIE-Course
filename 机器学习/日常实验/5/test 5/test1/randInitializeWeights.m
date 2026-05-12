function w = randInitializeWeights(layers)
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));
efun = @(x, y) sqrt(6)./(sqrt(x + y));
epsilon_init = repelem(bsxfun(efun, layers(1:end-1), layers(2:end)), nel);
w = (2*rand(sum(nel), 1) - 1).*epsilon_init';
end
