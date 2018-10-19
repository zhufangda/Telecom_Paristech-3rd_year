function [] = show_delta(delta, buffer)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    ncol = size(delta,1)
    figure()
    yyaxis right
    plot(delta(:,1))
    hold on
    yyaxis left
    plot(buffer(:,1))
    legend({'$w_1$', '$\Delta w_1$'},'Interpreter','latex');

    figure()
    yyaxis right
    plot(delta(:,2))
    hold on
    yyaxis left
    plot(buffer(:,2))
    legend({'$-b$', '$\Delta b$'},'Interpreter','latex');
end


