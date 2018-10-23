function [ likehood, likehood_log ] = calc_vraisemblance(phrase, trans_matrix, dict)
% calculate  likelihood function for a sentence specified by the
% arguements.
%   Input : 
%       phrase : a string to calculate the likehood.
%       trans_matrix : the transition matrix
%       dict : a dictionary. The key is the state, the value is the letter.
%   Output :
%       likehood : the likehoor for the fonction
%       likehood_log : the log(likehood) in order to get result more
%       exact.


% preprocessing of text
phrase = lower(strtrim(phrase));
phrase = ['-' phrase];
phrase = strrep(phrase,' ','+-');
phrase = strrep(phrase,'.','+.');

seq_size = length(phrase);
state_seq = zeros(1,seq_size);

for i = 1: seq_size
    if(phrase(i) == '-')
        state = 1;
    elseif(phrase(i) == '+')
        state = 28;
    else
        state = find(strcmp(phrase(i), dict(:,2)),1);
    end
    state_seq(1,i) = state;
end

likehood = 1;
likehood_log = 0;
for i = 2:seq_size
    likehood = likehood * trans_matrix(state_seq(i-1),state_seq(i) ); 
    likehood_log = likehood_log + log(trans_matrix(state_seq(i-1),state_seq(i) ));
end

