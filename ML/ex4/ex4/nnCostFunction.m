function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%TRANSFORM y into a matrix fo size m x num_labels with boolean categorical rows
Y = zeros(m,num_labels);
#Y(:,1) = [y == 10];
for i = 1:(num_labels)
  Y(:,i) = [y == i];
end

#----- implement FWD-propagation for computing J-----------
#first activation level values INPUT LAYER 
a1 = [ones(m,1), X]'; 
# second activation values HIDDEN LAYER
z2 = Theta1 * a1;
a2 = [ones(1,m);sigmoid(z2)];
# third activation values OUTPUT LAYER
z3 = Theta2 * a2;
a3 = sigmoid(z3);



#------- compute J ---------------------
# get the values of J summing over the cols
J = -1/m * sum(sum(  Y' .* log(a3) + (1-Y') .* log(1 - a3)  ));

# add the regularization term to J
regterm = lambda/(2*m) *( sum(sum(Theta1(:,2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)) );
J = J + regterm;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

d3 = a3 - Y';
d2 = Theta2' * d3 .* [ones(1,m); sigmoidGradient(z2)]; 
# remove d2's bias unit
d2 = d2(2:size(d2,1),:);

# ACCUMULATE the errors to form the gradient
Delta2 = d3 * (a2)';%, dim = 1);
Delta1 = d2 * (a1)';
grad = 1/m * [Delta1(:) ; Delta2(:)];
% -------------------------------------------------------------

% =========================================================================
% ADD REGULATIZATION term to the graidents
regterm2 = [zeros(num_labels,1),(lambda*Theta2)(:,2:(hidden_layer_size+1)) ];
regDelta2 = Delta2 + regterm2;
regterm1 = [zeros(hidden_layer_size,1), lambda*(Theta1)(:,2:input_layer_size+1)];
regDelta1 = Delta1 + regterm1;
#regDelta1 = Delta1 + lambda*(Theta1);

grad = 1/m * [regDelta1(:) ; regDelta2(:)];




%TRANSFORM y into a matrix fo size m x num_labels with boolean categorical rows
%and PREDICT the values of h_theta
%[maxval, p] = max(a3,[],dim=2);
%h_theta = zeros(m,num_labels);
%h_theta(:,1) = [p == 10];
%for i = 1:(num_labels-1)
%  h_theta(:,i+1) = [p == i];
%end


end
