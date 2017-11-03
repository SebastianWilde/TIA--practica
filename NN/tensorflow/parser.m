data = load('new_car.txt');
y = data(:,7);
y_new = eye(4)(y,:);
data_test = load('new_car_prueba.txt');
y_test = data_test(:,7);
y_new_test = eye(4)(y_test,:);

n_col = size(y_new,2);

tex = fopen('new_car_y.txt','wt');
for i = 1:size(y_new,1)
	fprintf(tex,'%d,',y_new(i,1:n_col-1))
	fprintf(tex,'%d',y_new(i,n_col))
	fprintf(tex,'\n')
end
fclose(tex)

n_col = size(y_new_test,2);

tex = fopen('new_car_test_y.txt','wt');
for i = 1:size(y_new_test,1)
	fprintf(tex,'%d,',y_new_test(i,1:n_col-1))
	fprintf(tex,'%d',y_new_test(i,n_col))
	fprintf(tex,'\n')
end
fclose(tex)