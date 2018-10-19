function [] = show_images( images, names, shape)
% Show all image in a single figure
%   Input : 
%       images : The images list, the last index reprensent the image
%           numbre
%       names : The list of figure name
%       shape : The shape of subplot

figure()
nr = shape(1);
nc = shape(2);
len = size(images, 3);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
for i = 1:len
    subplot(nr,nc,i) , image(images(:,:,i));
    title(char(names{i})); 
    colormap(gray(256));
    daspect([1 1 1]);
end

end

