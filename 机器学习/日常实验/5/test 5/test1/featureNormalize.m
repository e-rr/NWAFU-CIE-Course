function [X_norm, mu, sigma] = featureNormalize(X)
mu = mean(X);
sigma = std(X);
X_norm = (X - repmat(mu,size(X, 1), 1))./repmat(sigma,size(X, 1), 1);
X_norm(isnan(X_norm)) = 0;

end
