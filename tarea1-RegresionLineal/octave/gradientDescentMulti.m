function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp_X = X;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%La hipotesis es la transpuesta de theta por X
    %aux = X*theta - y;
    for i = 1:m,
        temp_X(i,:) = (X(i,:)*theta-y(i))*X(i,:);
    end
    theta = theta - (alpha/m) * sum(temp_X)';
        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCostMulti(X, y, theta);

end

end
