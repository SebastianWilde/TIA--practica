function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%Algunos posibles valores para C y sigma
%posibles_valores = [0.01,0.03,0.1,0.3,1,3,10,30];
posibles_valores = [0.01,0.1,1];
%Promedio de predicciones malas
error_inicial = 1.0;

for nuevo_C = posibles_valores,
	for nuevo_sigma = posibles_valores,
		model = svmTrain(X,y,nuevo_C,@(x1,x2)gaussianKernel(x1,x2,nuevo_sigma));
		prediccion = svmPredict(model,Xval);
		nuevo_error = mean(double(prediccion~= yval));
		%Si mi nueva error, es mas pequeño que el anterior, se actualiza los datos
		if error_inicial > nuevo_error ,
			error_inicial = nuevo_error;
			C = nuevo_C;
			sigma = nuevo_sigma;
		end
	end
end

% =========================================================================

end
