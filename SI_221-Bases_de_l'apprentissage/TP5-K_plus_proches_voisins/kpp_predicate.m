function [ y_test ] = kpp_predicate( X_test , X_train, y_train, k)
% Returns the result of the kpp
%   Input :
%       X_test : the test vector
%       X_train : the training set 
%       y_train : the class of each sample in the training set
%       k : number of the neighbor
%   Output :
%       y_test : the result of predicate for the x_test
%       

    train_row = size(X_train,1); 
    test_row = size(X_test,1);
    y_test = zeros(test_row,1);


    for i = 1:test_row
        x_matrix = repmat(X_test(i, :), train_row, 1);
        dist = sqrt(sum((x_matrix - X_train).^2,2));
        [B, I] = mink(dist, k);
        y_test(i,1) = mode(y_train(I));

    end
end

