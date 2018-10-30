function [ conf_max ] = confusion_matrix( y_val, y_predicate)
% Get confusion matrix

nb = 10;
nb_sample = size(y_val,1);
conf_max = zeros(nb, nb);
    for i = 1:nb_sample
        conf_max(y_val(i), y_predicate(i)) = conf_max(y_val(i), y_predicate(i)) + 1;
    end
    
end

