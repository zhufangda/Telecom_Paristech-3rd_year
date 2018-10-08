function  Z = normpdf(X,Y, mu, sigma)
    nb = size(X,2);
    Z = zeros(nb, nb);
   
    for i = 1:nb
        for j = 1:nb
            input = [X(i), Y(j)];
            tmp1 = exp(-0.5 * (input - mu) * inv(sigma) * (input- mu).');
            tmp2 = sqrt(4*pi*pi*det(sigma));
            Z(i,j) = tmp1/tmp2;
        end
    end
end
