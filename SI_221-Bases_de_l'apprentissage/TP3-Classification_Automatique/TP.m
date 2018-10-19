%% SI 221 : ACP et k-moyennes

%% 1 Description des donn�es
%% 1.1 Donn�es LANDSAT sur Tarascon
[tarascons, tara_names] = load_images('LANDSAT/', [512, 512]);
show_images(tarascons, tara_names, [3,3]);

%% 1.2 Donn�ees SPOT sur Tarascon

[spots, spot_names] = load_images('SPOT/', [512, 512]);
show_images(spots, spot_names, [1,3]);

spots_clear = image_clear(spots);
show_images(spots_clear, spot_names, [1,3]);



%% 1.3 Donn�es LANDSAT sur Kedougou

[kedougous, kedo_names] = load_images('Landstat_Kedougou/', [256, 256]);
show_images(kedougous, kedo_names, [3,3]);

kedougous_clear = image_clear(kedougous);
show_images(kedougous_clear, kedo_names, [3,3]);

%% 3 Application aux composantes de LANDSAT

nb_feature = 8;
nbPix = 512*512;

tabimage = reshape(tarascons(:,:, 1:nb_feature), [nbPix, nb_feature]);
[T, W, D] = my_PCA(tabimage);

post_imgs = reshape(T, 512,512,8);
%% Alors on obtiens le pourcentage d'inertie associ� � chaque image.
lambda = D / sum(D);
lambda'

%% 
% On costacte que les pourcentages d'inertie de trois premier canal sont tr�s petite, 
% le rest  ont un pourcentage important. Cela bien correspond � notre intuition. En effet, 
% le coleur de la terre et celui de la rivi�re sur l'image de canal 1 sont 
% tr�s proches , il est difficile de distinguer le fronti�re.En revanche, 
% on peut trouver le fronti�re plus facile dans l'image de canal 8. 
% Il est �vident que le canal 8 contiens plus d'information. 

for i = 1:8
    M = post_imgs(:,:,i);
    I = 255 * (M - min(M(:)))/(max(M(:)) - min(M(:)));
    subplot(3,3,i) , image(I); 
    title( strcat(['$\lambda_' num2str(i)  '=' num2str(lambda(i)) '$']), 'Interpreter', 'latex'); 
    colormap(gray(256))
    daspect([1 1 1]);
end

%%
% Afin de conserver au mois de 95% de l'information, il suffit de garder 4
% images, c'est � dire canal5, canal6 ,canal7 ,canal8 .
cumsum(sort(lambda, 'descend'))'

%% 4. Classification automatique : algorithme des K-moyennes
% Afin de trouver les prototypes pour chaque canal, on tracer
% l'histograme.

figure()
for i = 1:8
    img_tmp = tarascons(:,:,i);
    subplot(3,3,i),hist(img_tmp(:), 0:1:255);
        xlim([0,255]);
end

%% 
% A partir des figures ci-dessous, on choisir les points ci-dessus comme les prototypes initials.
%
proto1 = [81,95];
proto2 = [66,90];
proto3 = [52,62];
proto4 = [18,90];
proto5 = [16,85];
proto6 = [13,85];
proto7 = [12,30];
proto8 = [28,64];


tabproto = [proto1; proto2; proto3; proto4; proto5; proto6; proto7; proto8];
tabimage = reshape(tarascons(:, :, 1:8), [512*512, 8]);
classes = zeros(512*512, 8);
for i=1:8
    classes(:, i) = kmeans2(tabimage(:,i), tabproto(i,:)');
end

Ms = reshape(classes, 512,512,8);
Is = image_clear(Ms);
show_images(Is, tara_names, [3,3]);
%%
% D'apr�s la figure ci-dessus, kmeans n'est pas capable de distinguer 
% la terre et la rivi�re pour le trois premier.En effet, le coulour de rivi�re 
% et un partie des pixel sur sol sont proche. Donc des pixels sur sol sont consid�re
% comme pixel sur rivi�re. Cela correspond bien aux resultats dans la partie de PCA.


%% 5.BONUS : ACP et k-moyennes sur de nouvelles donn�es
%% 5.1 Application aux composantes XS de SPOT
nbFeature = 3;
nbPix = 512*512;

tabimage = reshape(spots, [nbPix, nbFeature]);
[T, V, D] = my_PCA(tabimage);

post_imgs = reshape(T, 512,512,3);
%% Alors on obtiens le pourcentage d'inertie associ� � chaque image.
lambda = D / sum(D);
lambda'

%% 
% On costacte que les pourcentages d'inertie de trois canal est important.
% Il garde plus de d�tail. Mais il est plus facile de distinguer la
% fronti�re entre le rivi�re et la terre. Donc le deux canal est plus utils.

for i = 1:nbFeature
    M = post_imgs(:,:,i);
    I = 255 * (M - min(M(:)))/(max(M(:)) - min(M(:)));
    subplot(1,3,i) , image(I); 
    title( strcat(['$\lambda_' num2str(i)  '=' num2str(lambda(i)) '$']), 'Interpreter', 'latex'); 
    colormap(gray(256))
    daspect([1 1 1]);
end

%%
% Afin de conserver au mois de 90% de l'information, il suffit de garder 2
% images, c'est � dire les deux derniers images .
cumsum(sort(lambda, 'descend'))'


%% 5.2 Application aux composantes de LANDSAT sur Kedougou
nb_feature = 7;
nbPix = 256*256;

tabimage = reshape(kedougous, [nbPix, nb_feature]);
[T, W, D] = my_PCA(tabimage);

post_imgs = reshape(T, 256,256,7);
%% 
% Alors on obtiens le pourcentage d'inertie associ� � chaque image.
lambda = D / sum(D);
lambda'

%% 
% Pour l'image 2,3,4, m�me leurs valeurs propres sont faible, mais on peut
% observer bien la contour de rivi�re. le canal 4 et canal 5 sont sensible
% pour  l'altitude.  Donc on ne peut pas d�cider simplement la importance
% de chaque feature par valeur propre.

for i = 1:7
    M = post_imgs(:,:,i);
    I = 255 * (M - min(M(:)))/(max(M(:)) - min(M(:)));
    subplot(3,3,i) , image(I); 
    title( strcat(['$\lambda_' num2str(i)  '=' num2str(lambda(i)) '$']), 'Interpreter', 'latex'); 
    colormap(gray(256))
    daspect([1 1 1]);
end

%%
% Afin de conserver au mois de 95% de l'information, il suffit de garder 4
% images, c'est � dire canal5, canal6 ,canal7 ,canal8 .
sort(lambda, 'descend')'
cumsum(sort(lambda, 'descend'))'


%% 5.3 k-moyennes sur SPOT et LANDSAT-Kedougou
% Si on utilse tous les canaux sur  landsat-kedougou, on obtiens le
% resultat ci-dessous


figure()
for i = 1:6
    img_tmp = kedougous(:,:,i);
    subplot(3,2,i),hist(img_tmp(:), 0:1:255);
    xlim([0,255]);
end

dataset = reshape(kedougous(:, :,1:6), [256*256, 6]);
index = randsample(1:length(dataset), 5);
proto = dataset(index, :);


[classe, nb_iter] = my_kmeans(dataset, proto);

showClass(classe,[256 256]);

figure();

gplotmatrix(dataset, dataset, classe);

rates = zeros(1,5);
for i = 1:5
    rates(1,i) = sum( classe==i );
end

rates = rates / size(dataset,1)

