function [post_table, V, D ] = my_PCA( data )
% Implementation of ACP
% Input:
%     Data : The input data. The each row represent the a sample.
%            Each column represent a feature.
% Output:
%       post_table : Matrix consisting of n row vectors, 
%           where each vector is the projection of the corresponding 
%           data vector from matrix data onto the basis vectors 
%           contained in the columns of matrix V.
%       V : Matrix of basis vectors
%       D : A row vector. Each element reprensente the inertia percentages
%           for corresponding the feature column. 

nbSample = size(data, 1);

%% centrage et réduction des données
means = repmat(mean(data), [nbSample, 1]);
sigma = repmat(sqrt(var(data)), [nbSample, 1]);
reduce_tab = (data - means) ./ sigma;

%% calcul de la matrice de covariance des donn´ees centr´ees r´eduites ;
covs = cov(reduce_tab);

%% calcul des valeurs propres ?j et vecteurs propres uj de la matrice de covariance ;
[V, D] = eig(covs);
D = diag(D);

%% calcul des composantes principales
post_table = reduce_tab * V;

end

