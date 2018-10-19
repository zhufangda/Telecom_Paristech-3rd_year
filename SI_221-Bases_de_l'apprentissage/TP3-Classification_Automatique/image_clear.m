function [ image_clear ] = image_clear(images)
% Augment the image contrast
%   Input :
%       images : The image list to traite
%   Output :
%       images_clear : the traited image list
nb = size(images, 3);
image_clear = images;
for i = 1:nb;
    im = images(:,:,i);
    image_clear(:,:,i) = 255 * ( im - min(im(:)))/(max(im(:)) - min(im(:)));
end


