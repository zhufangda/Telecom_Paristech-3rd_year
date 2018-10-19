function [ images, names] = load_images( path, dim )
% Load all image from a path

files = dir([ path '/*.dim']);
nbFiles = size(files, 1);
images = ones(dim(1),dim(2), nbFiles);
names = cell(1,nbFiles);

for i = 1:nbFiles
    name = strsplit(files(i).name,'.');
    names{i} = name(1);
    images(:,:,i) = ima2mat(char(strcat(path , name(1))));
end

end

