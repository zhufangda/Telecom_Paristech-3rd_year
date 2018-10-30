function [ B I ] = mink( vector, k )
% Returns the k largest values and their index
%   Input :
%       list : a vector
%       K : the number of the largest element
%   Output :
%       B : the K largest value in the vector
%       I : the index of the K leargest

    [dis, index] = sort(vector);
    B = dis(1:k);
    I = index(1:k);

end

