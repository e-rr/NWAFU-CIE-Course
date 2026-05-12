function [ Theta ] = unpackTheta( theta, layers )
nl = numel(layers) - 1; 
Theta = cell(nl, 1); 
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));
for ii=1:nl
    Theta{ii} = reshape(theta(1:nel(ii)), layers(ii+1), layers(ii) + 1);
    theta(1:nel(ii)) = [];
end
end

