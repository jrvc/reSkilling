function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% the cost function is the squared error function, i.e.,

%X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
%theta = zeros(2, 1);

%since theta = [0,0]' our hypothesis is h_theta = 0
h_theta = X*theta; % the evaluation of all the observation for h_theta(x)= theta(1) + theta(2)*x
J = 1/(2*m)*sum((h_theta - y).^2);



% =========================================================================

end
