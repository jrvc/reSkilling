function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


[J, grad] = costFunction(theta, X, y) ;
J = J + lambda/(2*m) * sum(theta(2:length(theta)) .^2) ;
grad = grad + [0; lambda/m * theta(2:length(theta))];



% =============================================================

end
