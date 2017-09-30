%% Machine Learning Online Class - Exercise 2: Logistic Regression con car-data
%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
%% Initialization
clear ; close all; clc
data = load('new_car.txt');
X = data(:, [1:6]); y = data(:, 7);
%%One vs all
y_unacc = y_acc = y_good = y_vgood = y;
y_unacc(y_unacc ~= 1) = 0;

y_acc(y_acc == 2) = 1;
y_acc(y_acc ~= 2) = 0;

y_good(y_good == 3) = 1;
y_good(y_good ~= 3) = 0;

y_vgood(y_vgood == 4) = 1;
y_vgood(y_vgood ~= 4) = 0;

%%obteniendo los thetas
theta_unacc = ex2_reg(X,y_unacc);
theta_acc = ex2_reg(X,y_acc);
theta_good = ex2_reg(X,y_good);
theta_vgood = ex2_reg(X,y_vgood);

%%Etapa de prueba
data_test = load('new_car_prueba.txt');
X_test = data_test(:, [1:6]); y_test = data_test(:, 7);
%[m, n] = size(X_test);
X_test = mapFeature(X_test(:,1), X_test(:,2),X_test(:,3),X_test(:,4),X_test(:,5),X_test(:,6));

y_test_unacc = y_test_acc = y_test_good = y_test_vgood = y_test;

y_test_unacc(y_test_unacc ~= 1) = 0;

y_test_acc(y_test_acc == 2) = 1;
y_test_acc(y_test_acc ~= 2) = 0;

y_test_good(y_test_good == 3) = 1;
y_test_good(y_test_good ~= 3) = 0;

y_test_vgood(y_test_vgood == 4) = 1;
y_test_vgood(y_test_vgood ~= 4) = 0;

%% ============== Part 4: Predict and Accuracies ==============

%prob = sigmoid((1 45 85) * theta);
%fprintf(('For a student with scores 45 and 85, we predict an admission ' ...
%         'probability of %f\n\n'), prob);

% Compute accuracy on our training set
p1 = predict(theta_unacc, X_test);
fprintf('Train Accuracy para unacc: %f\n', mean(double(p1 == y_test_unacc)) * 100);
p2 = predict(theta_acc, X_test);
fprintf('Train Accuracy para acc: %f\n', mean(double(p2 == y_test_acc)) * 100);
p3 = predict(theta_good, X_test);
fprintf('Train Accuracy para good: %f\n', mean(double(p3 == y_test_good)) * 100);
p4 = predict(theta_vgood, X_test);
fprintf('Train Accuracy para vgood: %f\n', mean(double(p4 == y_test_vgood)) * 100);