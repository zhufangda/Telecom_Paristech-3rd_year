function [ next_state ] = etat_suivant(current_state, trans_matrix )
% Get next state from current state.
%   Input:
%       current_state : current state
%       trans_matrix : transition matrix
%   Output:
%       next_state : the plus possible state known the current state.

unif = rand();
cs = cumsum(trans_matrix(current_state,:));
next_state = 1;

while (unif >= cs(next_state))
    next_state = next_state + 1;
end

