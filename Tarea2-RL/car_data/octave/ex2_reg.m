%% Machine Learning Online Class - Exercise 2: Logistic Regression
function theta = ex2_reg(X,y)
X = mapFeature(X(:,1), X(:,2),X(:,3),X(:,4),X(:,5),X(:,6));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
fprintf('theta: \n');
fprintf(' %f \n', theta);

end


