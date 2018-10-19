function [  ] = showClass( classes, dim )
% Show the result of classification
% Input:
%   classes : result of classification
%   dim : shape of the image

M = reshape(classes, dim(1),dim(2));
figure()
image(M);
colormap(prism);

end

