function X = mapFeature(X)   
    out = ones(size(X(:,1))) ;
    [m,n] = size(out);
    x = ones(size(m,1));
    for i = 1:m
          x(i,1) = X(i,6)*X(i,6);
    end
    X = [X x];
    for i = 1:m
          x(i,1) = X(i,6)*X(i,4);
    end
    X = [X x];
        for i = 1:m
          x(i,1) = X(i,4)*X(i,4);
    end
    X = [X x];

end