function [W, delta, buffer, k] = perceptron(samples, classes,eta ,n_iter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    W = ones(2,1);
    buffer = zeros(n_iter * size(samples,1),2);
    delta = zeros(n_iter * size(samples,1),2);
    taille = size(samples,1);
    as = size(n_iter * size(samples,1),1);
      
    W_ep = ones(2,1);
    
    for k = 1:n_iter 
        for i= 1: taille
            Y = [samples(i);1];
            a = sign( 2*((W.' * Y)>=0)-1);
            as((k-1)*taille + i,:) = a;
            W_ = W  - (a - classes(i)) * eta * Y;
            buffer((k-1)*taille + i,:) = W_;
            delta((k-1)*taille + i,:) = abs(W_ - W).';
            W = W_;
        %     if(delta((k-1)*taille + i,1) == 0 && delta((k-1)*taille + i,2) == 0)
        %        break 
        %     end

        end
        delta_ep = W - W_ep; 
        if(delta_ep(1)== 0 && delta_ep(2)== 0)
            break;
        end
        W_ep = W;
    end

end

