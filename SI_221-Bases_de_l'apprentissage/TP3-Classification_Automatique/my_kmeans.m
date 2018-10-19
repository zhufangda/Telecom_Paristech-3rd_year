function [ classe, n_iter] = my_kmeans( data, init_proto)
% Implementation of native k means algo
%   Input:
%       data : the data to classify
%       init_proto : The initial centroid for the classification
%  output:
%       classe : The result of classification
%       n_iter : numbre of iteration

proto = init_proto;
old_proto = init_proto  - 1;
nbType = size(init_proto, 1);
n_iter = 0;
while(old_proto ~= proto)
    n_iter = n_iter+1 ;
    classe = kmeans2(data, proto);
    old_proto = proto;
    
    for i = 1:nbType
        proto (i,:) = mean(data(classe == i,:));
    end
end

