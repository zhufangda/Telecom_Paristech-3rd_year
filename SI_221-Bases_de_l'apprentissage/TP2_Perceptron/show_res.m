function [W] = show_res( W, samples, classes, x_lim, y_lim )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    figure()
    taille = size(samples,1);
    n = (x_lim(2)-x_lim(1)) *2 + 1;
    x = ones(2,n);
    x(1,:) = linspace(x_lim(1),x_lim(2), n).';
    y_line = W.' * x;
    y = (2*((W.' * x) >=0) - 1);
   
    plot(x(1,:), y_line);
    hold on
    plot(x(1,:), y)
    hold on
 scatter(samples, classes)
    xlim(x_lim);
    ylim(y_lim);
    legend({'$y=W^TX$', '$y = sign(W^TX)$', '$samples$'},'Interpreter','latex')




    accuracy = 0;
    for i=1:size(classes)
        X = [samples(i);1];
        y_predit = sign( 2*((W.' * X)>=0)-1);
        if(classes(i) == y_predit)
           accuracy = accuracy + 1 ;
        else
          %fprintf('index %d, value %d, classes %d, prediction %d', i, samples(i),classes(i), y_predit);
        end
    end
    fprintf('Accuracy rate %f, Error rate %f\n', accuracy / taille, 1.0 - accuracy / taille);

end


