function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% initialize to zero, to return anything 
J = 0;
grad = zeros(size(theta));

% ============================================
% Compute the cost of a particular choice of theta.

%recall : h_theta = 1 / (1 + e^(-theta'*x)); 
J =-1/m * sum(y .* log(sigmoid(X*theta)) + (1-y) .* log(1 - sigmoid(X*theta)));
% Compute the partial derivatives.  
% grad = partial derivatives of the cost w.r.t. each parameter in theta
grad = 1/m * X'*(sigmoid(X*theta) - y);

% ==============================================

end
