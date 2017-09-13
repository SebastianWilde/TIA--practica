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
%El costo viene dado por la formula J(theta0,theta1) = 1/(2m) sum_i=1:m
%(hipotesis - el dato)^2
J = sum(theta(1)+ (X(:,2).*theta(2) - y).^2);
J = (1/(m*2))*J;
% =========================================================================

end
