%% Travaux Pratique - Chaines de Morkov
% ZHU Fangda

%% 2. Cha�ne de Markov



eval('correspondance')

%% 2.a. Matrice de transitions
load 'bigramenglish'
load 'bigramfrancais'

%%
% La premier ligne correspond aux propabilit�s de l'occurence au d�but de mot 
% pour chaque lettre. La dernier colonnes correspond aux probabilit�s de l'occurence � la
% fin du mot pour chaque lettre de l'alphabet.
% On peut afficher la transition la plus fr�quente depuis chaque lettre avec m�thode ci-dessous:

[value, index] = max(bigramenglish');

a = corresp(index + 28);

%%
% La resultat indique que la lettre 't' est le plus fr�quente lettre comme
% la premier lettre du mot. 

display(a)


%%  2.b G�n�rer un mot
% 
% L'impl�mentation du fonction *etat_suivant*
% 
% <include>etat_suivant.m</include>
% 
% L'impl�mentation du fonction *genere_stat_seq*
% 
% <include>genere_state_seq.m</include>
% 
% L'impl�mentation du fonction *display_seq*
% 
% <include>display_seq.m</include>
% 

a = genere_state_seq(bigramenglish);
seq = display_seq(a, corresp);

%%
%
bar(1:28,bigramenglish(5,:));

%% 
% On peut imagine que l'on jette un boule dans le zone blue al�atoirement, 
% si on jette dans le zone de lettre a, alore on reprise 'a' comme l'�tat
% suivant. Comme la proportion surface de la zone de 'a' sur la surface total
% bien correspond � son probabilit�. Avec cette methode, on peut tranformer
% la distribution uniform�ment � la distribution d�cris par la matrix.

%% 3. G�n�rer une phrase
%
% L'impl�mentation du fonction *modifie_mat_dic*
% 
% <include>modifie_mat_dic.m</include>
% 
[dict, trans_matrix] = modifie_mat_dic(corresp, bigramenglish);
a = genere_state_seq(trans_matrix);
seq = display_seq(a, dict);


%% 4. Reconnaissance de la langue
% 
% L'impl�mentation du fonction *modifie_mat_dic*
% 
% <include>calc_vraisemblance.m</include>
% 

[dict, trans_matrix_en] = modifie_mat_dic(corresp, bigramenglish);

[dict, trans_matrix_fr] = modifie_mat_dic(corresp, bigramfrancais);

[pro_en, pro_log_en] = calc_vraisemblance('to be or not to be.', trans_matrix_en, dict)
[pro_fr, pro_log_fr] = calc_vraisemblance('to be or not to be.', trans_matrix_fr, dict)

display(strcat(['The likehood for the English is ' num2str(pro_en) ]));
display(strcat(['The likehood for the French is ' num2str(pro_fr) ]));

%%
% D'apr�s la vraisemblance, ce phrase est plit�t anglais.
%%

[dict, trans_matrix_en] = modifie_mat_dic(corresp, bigramenglish);

[dict, trans_matrix_fr] = modifie_mat_dic(corresp, bigramfrancais);

[pro_en, pro_log_en] = calc_vraisemblance('etre ou ne pas etre.', trans_matrix_en, dict)
[pro_fr, pro_log_fr] = calc_vraisemblance('etre ou ne pas etre.', trans_matrix_fr, dict)

display(strcat(['The likehood for the English is ' num2str(pro_en) ]));
display(strcat(['The likehood for the French is ' num2str(pro_fr) ]));

%%
% D'apr�s la vraisemblance, ce phrase est plit�t anglais.
