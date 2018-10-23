function [ corresp_post, trans_mat_post] = modifie_mat_dic(corresp_ori, trans_mat_ori )
% Transform the original transition matrix and dict.
%   Input :
%       corresp_ori : the original dictionary
%       trans_mat_ori : the original transition matrix
%   Output :
%       corresp_post : the dictionary after transformation
%       trans_mat_post : the original after transformation
%
    corresp_post = corresp_ori;
    corresp_post{29,2} = '.';
    corresp_post{29,1} = 29;
    trans_mat_post = trans_mat_ori;
    trans_mat_post(:,29) = 0;
    trans_mat_post(29,:) = 0;
    trans_mat_post(28,28) = 0.0;
    trans_mat_post(28, 1) = 0.9;
    trans_mat_post(28, 29) = 0.1;
    trans_mat_post(29,29) = 1;
end

