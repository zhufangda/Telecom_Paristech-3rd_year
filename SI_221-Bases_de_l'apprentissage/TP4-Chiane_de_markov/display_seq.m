function [ letter_seq ] = display_seq( index_seq, corresp )
% Transform a sequence of state to a sequence of letter.
%   Input : 
%       index_seq : sequence of state
%       corresp : a dictionary.
%                   The key is the state , and the value is the letter. 
%   Output :
%       lettrer_seq : sequence of lettre

letter_seq = [];
for i = 1:size(index_seq,2)
    state = index_seq(i);
    if(state==1 && index_seq(i-1)==28)
        letter_seq = [letter_seq ' '];
    elseif(state == 28)
        continue;
    else
        letter_seq = [letter_seq corresp{index_seq(i),2}] ;
    end
end

letter_seq = char(letter_seq);
display(letter_seq);
