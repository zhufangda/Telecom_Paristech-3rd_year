%% TP kppv : k plus proches voisins
% Author : ZHU Fangda

%% Les donn�es
%

close all;
load data_app;
X_train = x; y_train = S;
load data_test;
X_test = x; y_test = S;

m = 16;
im = reshape(x(m,:), 28, 28)';
image(255*ones(28,28) - im);
colormap(gray);
S(m);



figure()
hist(Sa)
title('Distribution of class in the training set')

figure()
hist(S_val)
title('Distribution of class in the test set')

%%
% D'apr�s la figure ci-dessous, on peut constacte que les examples sont
% pr�sque �quir�parties suivant les classes.

%% 3 Classement par kpp
% On impl�mente l'algorithme de kpp comme ci-dessous:
% 
% <include>kpp_predicate.m</include>
%
% <include>mink.m</include>
%
%
% <include>confusion_matrix.m</include>
%
%
% <include>show_confusion_matrix.m</include>
%
k = 4;
for k = [1,3,4,5]
    y_pre = kpp_predicate(X_test, X_train, y_train, k);
    error_score = 1 - sum(y_test == y_pre) / size(y_test,1);
    str = sprintf('K = %d \t error score = %.2f', i, error_score);
    display(str);
    display(conf_matrix);
    conf_matrix = confusion_matrix(y_test, y_pre);
    show_confusion_matrix(conf_matrix, k)
    
end
%%
% D'apr�s la matrix de confusion, on peut trouver que la plus part des
% image sont bien class�e.

%% 4.1 1-ppv avec prototype
% 
%

prototype = zeros(10,784);
y_proto = zeros(10,1);
for i = 1:10
    y_proto(i) = i;
    prototype(i,:) = mean(X_train(y_train == i, :),1);
end

y_pre = kpp_predicate(X_test, prototype, y_proto, 1);

error_score = 1 - sum(y_test == y_pre) / size(y_test,1)

conf_mat_proto = confusion_matrix(y_test, y_pre);
title('The confusion matrix for the 1-ppv avec prototype')

%%
% D'apr�s la r�sultat ci-dessus, on constate que le taux de erreur est
% augement�. Mais le resultat ce qu'on obtiens avec prototype est assez
% bien. Cette methode fais mois de calcules. Avec cette m�thode, une fois on 
% obtiens le model, le model ne enregistre que le prototype. Du coupe, la 
% complexit� du temps et la compl�xit� de l'�sp�re est mieux que la model obtenus en 3. 
