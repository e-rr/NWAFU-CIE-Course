function g = sigmoidGradient(z)
sigmoid_z = sigmoid(z);
g = sigmoid_z.*(1 - sigmoid_z);

end
