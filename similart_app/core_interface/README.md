# COMMANDE POUR EXECUTER LE SERVEUR

Dans le dossier racine du projet `branches\test...\WebApp\ProjetCBIR\`
taper : `python manage.py runserver`

# IGNORER UN FICHIER/DOSSIER POUR LES COMMITS

Les datasets resteront en local car impossible à commit (trop lourd), donc il faut taper la commande : `svn propset svn:ignore 'image_dataset' .`
Pour voir tous les fichiers/admin : `svn proplist -v`

# POUR COMMIT

1. Ajouter les fichiers/dossiers que vous voulez rajouter au dépôt : `svn add *` ( '\*' pour TOUT, sinon remplacer par le nom du fichier)
2. Ensuite il faut commit : `svn commit -m "commit message"` avec un message de commit court et clair !

# POUR METTRE A JOUR SA BRANCHE

`svn update` par exemple si vous voulez récuperer le code que qlqn à écrit

# README - Projet Django

Ce README vise à fournir une compréhension d'un projet Django, en parlant de sa structure etc.

## Structure du Projet

## Pourquoi deux dossiers ?

La séparation en deux dossiers, `ProjetCBIR` et `app`, est une pratique courante dans les projets Django pour une meilleure organisation et une meilleure modularité.

- **ProjetCBIR/**: Ce dossier contient les paramètres et la configuration globale du projet. Il est utilisé pour définir les paramètres globaux qui s'appliquent à l'ensemble du projet Django, tels que les réglages généraux, les routes URL globales, etc.

- **app/**: Ce dossier contient l'application spécifique à notre projet, dans notre cas, une application CBIR. Il contient les modèles, les vues, les URL et d'autres composants spécifiques à cette application. Cette structure permet de mieux organiser le code en regroupant les fonctionnalités connexes dans des modules distincts et réutilisables.

- **ProjetCBIR/** (c'est la racine du projet, c'est le projet django en lui même, qui contient plusieurs sous-app comme notre app de cbir !)

  - **settings.py**: Ce fichier contient les paramètres de configuration globaux pour notre application Django. Il inclut des configurations telles que la base de données, les applications installées, les clés secrètes, etc. (par exemple, tailwind est considéré comme une app, donc on rajoute celui-ci aux app installées dans ce fichier)
  - **urls.py**: Ce fichier définit les routes URL pour l'ensemble du projet. Il associe les URL aux vues correspondantes dans différentes applications.

- **app/** (ça, c'est notre app principale cbir!)
  - **models.py**: Ce fichier définit les modèles de données pour notre application. Il contient des classes qui décrivent la structure des tables de la base de données.
  - **views.py**: Ce fichier contient les fonctions de vue pour notre application. Ces fonctions prennent en charge la logique de traitement des demandes et la génération des réponses.
  - **urls.py**: Ce fichier définit les routes URL spécifiques à notre application. Il associe les URL aux vues correspondantes définies dans `views.py`.
  - **forms.py**: Ce fichier contient les formulaires utilisés dans notre application Django. Les formulaires définissent la structure des données à saisir dans les formulaires HTML et effectuent la validation des données. Pour l'instant on l'utilise pas, mais pourquoi pas !

## Ajouter une Nouvelle Page

voici le processus étape par étape:

1. **Définition de l'URL**:

   - Ouvrez le fichier `app/urls.py`.
   - Ajoutez un nouveau chemin d'URL
   - Associez ce chemin à la vue correspondante à l'aide de `views.nom_de_la_vue`.

2. **Création de la vue**:

   - Ouvrez le fichier `app/views.py`.
   - Définissez une nouvelle fonction de vue qui gérera la logique pour cette page.
   - Cette fonction doit prendre au moins un argument `request` qui représente la demande HTTP entrante.

3. **Traitement de la demande**:

   - Dans la vue nouvellement créée, traitez la demande HTTP entrante selon les besoins de la page.
   - Ceci peut inclure le rendu d'un modèle HTML, l'extraction de données de la base de données, le traitement de formulaires, etc.

4. **Rendu de la réponse**:

   - Renvoyez une réponse HTTP appropriée depuis la vue.
   - Cela peut être un modèle HTML rendu à l'aide de `render()` ou une réponse JSON à l'aide de `JsonResponse()`.

5. **Mise à jour des modèles HTML** (si nécessaire):

   - Créez ou mettez à jour les modèles HTML dans le dossier `app/templates/` pour refléter la nouvelle page.
   - Utilisez les données fournies par la vue pour dynamiquement remplir et rendre le contenu de la page.

6. **Test de la nouvelle page**:
   - Lancez le serveur de développement Django en utilisant la commande `python manage.py runserver`.
   - Accédez à l'URL de la nouvelle page dans votre navigateur pour vérifier son fonctionnement correct.
