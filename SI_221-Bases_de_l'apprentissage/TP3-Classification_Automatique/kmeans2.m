function classe = kmeans2( matlig, mspe)
%
% matlig est le tableau : chaque ligne represente un pixel multicanaux de
% l'image
% mspe est la matrice des prototypes : une ligne par prototype
nld = size(matlig,1);
normspe=diag(mspe*mspe')';

%
% On calcule le produit scalaire en chaque pixel, avec chaque prototype
% on multiplie le resultat par 2
ddd=2*matlig*mspe';
% La valeur a minimiser est alors tout simplement
dval=ones(nld,1)*normspe-ddd;
%
% en effet, en chaque pixel, on calcule la norme de (d-pro)
%
%pour utiliser les possibilités de Matlab, on ne calcule pas la norme
%du vecteur initial puisque c'est une constante ne dependant pas des protos !!
%
% l'ordre matlab min permet d'avoir non seulement la valeur min mais aussi
% l'indice correspondant
[matmin,classe]=min(dval');
% disp(sprintf('OK >> la matrice classe ( en sortie)  a %d ligners',  nld ));
% disp(' utilisez reshape pour reformatter les resultats en format image');
% disp('par exemple pour une image initiale 64x64 :matima=reshape(classe,64,64);  ');
% disp('si vous voulez avoir une matrice comme fonction indicatrice de classe (valeur 0 ou 1) :');
% disp('   proto3 = classe==3 ');
% disp('le vecteur colonne proto3 est nul sauf pour les pixels appartenant a la classe 3');
