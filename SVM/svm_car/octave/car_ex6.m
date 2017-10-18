%% Machine Learning Online Class - Exercise 2: Logistic Regression con car-data
%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
%% Initialization
data = load('new_car.txt');
X = data(:, [1:6]); y = data(:, 7);
%%One vs all
y_unacc = y;
for i=1:size(y_unacc,1),
	if y_unacc(i) ~= 1,
		y_unacc(i) = 0;
	end
end


y_acc = y;
for i=1:size(y_acc,1),
	if y_acc(i) == 2,
		y_acc(i) = 1;
	else
		y_acc(i) = 0;
	end
end


y_good  = y;
for i=1:size(y_good,1),
	if y_good(i) == 3,
		y_good(i) = 1;
	else
		y_good(i) = 0;
	end
end


y_vgood = y;
for i=1:size(y_vgood,1),
	if y_vgood(i) == 4,
		y_vgood(i) = 1;
	else
		y_vgood(i) = 0;
	end
end


%%Etapa de prueba
data_test = load('new_car_prueba.txt');
X_test = data_test(:, [1:6]); y_test = data_test(:, 7);

y_test_unacc = y_test;
for i=1:size(y_test_unacc,1),
	if y_test_unacc(i) ~= 1,
		y_test_unacc(i) = 0;
	end
end

y_test_acc = y_test;
for i=1:size(y_test_acc,1),
	if y_test_acc(i) == 2,
		y_test_acc(i) = 1;
	else
		y_test_acc(i) = 0;
	end
end

y_test_good = y_test;
for i=1:size(y_test_good,1),
	if y_test_good(i) == 3,
		y_test_good(i) = 1;
	else
		y_test_good(i) = 0;
	end
end

y_test_vgood = y_test;
for i=1:size(y_test_vgood,1),
	if y_test_vgood(i) == 4,
		y_test_vgood(i) = 1;
	else
		y_test_vgood(i) = 0;
	end
end


%obteniendo los modelos
fprintf('Obteniendo modelo para unacc \n');
t=cputime;
[C_unacc,sigma_unacc,model_unacc] = ex6(X,y_unacc,X_test,y_test_unacc);
fprintf('C: %f, sigma: %f \n',C_unacc,sigma_unacc);
printf('Total cpu time: %f seconds\n', cputime-t);

fprintf('Obteniendo modelo para acc \n');
t=cputime;
[C_acc,sigma_acc,model_acc] = ex6(X,y_acc,X_test,y_test_acc);
fprintf('C: %f, sigma: %f \n',C_acc,sigma_acc);
printf('Total cpu time: %f seconds\n', cputime-t);

fprintf('Obteniendo modelo para good \n');
t=cputime;
[C_good,sigma_good,model_good] = ex6(X,y_good,X_test,y_test_good);
fprintf('C: %f, sigma: %f \n',C_good,sigma_good);
printf('Total cpu time: %f seconds\n', cputime-t);

fprintf('Obteniendo modelo para vgood \n');
t=cputime;
[C_vgood,sigma_vgood,model_vgood] = ex6(X,y_vgood,X_test,y_test_vgood);
fprintf('C: %f, sigma: %f \n',C_vgood,sigma_vgood);
printf('Total cpu time: %f seconds\n', cputime-t);
% %% ============== Part 4: Predict and Accuracies ==============

% % Compute accuracy on our training set
p1 = svmPredict(model_unacc,X_test);
fprintf('Train Accuracy para unacc: %f\n', mean(double(p1 == y_test_unacc)) * 100);
p2 = svmPredict(model_acc,X_test);
fprintf('Train Accuracy para acc: %f\n', mean(double(p2 == y_test_acc)) * 100);
p3 = svmPredict(model_good,X_test);
fprintf('Train Accuracy para good: %f\n', mean(double(p3 == y_test_good)) * 100);
p4 = svmPredict(model_vgood,X_test);
fprintf('Train Accuracy para vgood: %f\n', mean(double(p4 == y_test_vgood)) * 100);