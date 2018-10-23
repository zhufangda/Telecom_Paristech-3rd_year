function [ seq ] = genere_state_seq(trans_matrix)
% Generate a sequence of letter by transition matrix.
%   Input:
%       trans_matrix : The transition matrix for the HMM
%   Output:
%       seq :  index sequence of lettre

state = 1;
seq = [];
end_flag = size(trans_matrix, 1);
while(state ~= end_flag)
    state = etat_suivant(state, trans_matrix);
    seq = [seq state];
end




