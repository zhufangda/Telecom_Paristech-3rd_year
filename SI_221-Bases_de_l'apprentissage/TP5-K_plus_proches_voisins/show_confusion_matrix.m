function [  ] = show_confusion_matrix( matrix, k )
% Visualiez the confusion matrix
    norm = repmat(sum(matrix, 2),1,10);
    figure();
    imshow(1 - matrix ./ norm,'InitialMagnification','fit');
    title( strcat(['The confusion matrix for the k-ppv avec k=' ,  int2str(k)]) );    
    colormap('gray');
    axis on;
end

