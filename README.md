📊Prédiction du Churn Client 

Projet de Data Science et Machine Learning visant à analyser et prédire le départ des clients (Customer Churn) dans une entreprise de télécommunication.

 🎯 Objectif du projet

L'objectif de ce projet est de :

* comprendre les facteurs associés au départ des clients ;
* analyser les caractéristiques des clients ayant quitté l'entreprise ;
* identifier les clients présentant un risque élevé de churn ;
* construire un modèle de Machine Learning capable de prédire le départ des clients ;
* comparer plusieurs modèles de classification ;
* optimiser le modèle le plus performant.

📊 Dataset

Le projet utilise le dataset Telco Customer Churn.

Le dataset initial contient :

* 7 043 clients
* 50 variables

Les variables décrivent notamment :

* les caractéristiques démographiques des clients ;
* leur ancienneté ;
* les services souscrits ;
* leur type de contrat ;
* leurs informations de facturation ;
* leur niveau de satisfaction ;
* leur comportement de fidélisation ;
* leur statut de churn.

La variable cible utilisée pour la prédiction est :
ChurnLabel

Elle indique si le client a quitté l'entreprise :

* `Yes` → client ayant quitté l'entreprise ;
* `No` → client resté dans l'entreprise.

🧹 Nettoyage des données

Plusieurs étapes de préparation des données ont été réalisées :

* vérification des doublons ;
* traitement des valeurs manquantes ;
* nettoyage des variables catégorielles ;
* détection des variables constantes ;
* suppression des variables inutiles ;
* vérification des valeurs incohérentes ;
* détection des valeurs aberrantes avec la méthode IQR ;
* identification des variables pouvant provoquer une fuite de données.

Les variables suivantes ont notamment été supprimées :

CustomerID
Country
State
Quarter
City
ZipCode
Latitude
Longitude
Population

Après cette étape, le dataset contient :


7043 observations
41 variables


⚠️ Prévention de la fuite de données

Certaines variables du dataset sont directement liées au churn et ne doivent pas être utilisées comme variables explicatives.

Les variables suivantes sont donc exclues de la modélisation :


ChurnScore
ChurnCategory
ChurnReason
CustomerStatus


Cette étape permet d'éviter la fuite de données (Data Leakage) et d'obtenir une évaluation plus fiable des modèles.

 🔎 Analyse exploratoire

Une analyse exploratoire des données (EDA) est réalisée afin d'étudier les relations entre les caractéristiques des clients et le churn.

L'analyse porte notamment sur :

* la distribution du churn ;
* l'ancienneté des clients ;
* les types de contrats ;
* les charges mensuelles ;
* les services Internet ;
* les services supplémentaires ;
* la satisfaction des clients ;
* les recommandations ;
* les caractéristiques démographiques.

 🤖 Modèles de Machine Learning

Plusieurs modèles de classification sont utilisés afin de comparer leurs performances :

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors (KNN)

Les modèles sont évalués à l'aide de plusieurs métriques :

* Accuracy
* Precision
* Recall
* F1-score
* Matrice de confusion

 ⚙️ Prétraitement

Les données sont préparées avant l'entraînement des modèles.

Les principales étapes comprennent :

* séparation des variables explicatives et de la variable cible ;
* encodage des variables catégorielles ;
* normalisation des variables numériques ;
* séparation des données en ensemble d'entraînement et de test ;
* utilisation d'une séparation stratifiée afin de conserver la proportion des classes.

La séparation utilisée est :

80 % → entraînement
20 % → test
```

🔧 Optimisation

Le modèle le plus performant est ensuite optimisé à l'aide de GridSearchCV et d'une validation croisée en 5 parties.

Le F1-score est utilisé comme critère de sélection afin de rechercher un bon équilibre entre la précision et le rappel.

🏆 Résultats

Dans la configuration finale incluant la variable `SatisfactionScore`, le modèle Gradient Boosting obtient les meilleures performances.

Après optimisation, les résultats obtenus sur l'ensemble de test sont :

| Métrique  | Résultat |
| --------- | -------: |
| Accuracy  |  96.24 % |
| Precision |  96.25 % |
| Recall    |  89.30 % |
| F1-score  |  92.65 % |

Les meilleurs hyperparamètres obtenus pour le modèle Gradient Boosting sont :

n_estimators = 200
learning_rate = 0.1
max_depth = 3


 📌 Principales variables importantes

L'analyse de l'importance des variables montre notamment l'importance de :

* `SatisfactionScore`
* `TenureinMonths`
* `OnlineSecurity`
* `Number_of_Referrals`
* `InternetType`

Ces variables permettent de mieux comprendre certains facteurs associés au churn.

🛠️ Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Git
* GitHub

📁 Structure du projet

Churn_client/
│
├── app/
│   ├── avec_satisfaction/
│   │   └── app.py
│   │
│   └── sans_satisfaction/
│       └── app.py
│
├── data/
│   ├── raw/
│   │   └── TelcoCustomerChurn.csv
│   │
│   └── processed/
│       └── TelcoCustomerChurn_Clean.csv
│
├── models/
│   ├── avec_satisfaction/
│   │   ├── gradient_boosting_churn.pkl
│   │   ├── scaler.pkl
│   │   ├── frequency_encodings.pkl
│   │   └── model_information.pkl
│   │
│   └── sans_satisfaction/
│       ├── gradient_boosting_churn.pkl
│       ├── scaler.pkl
│       ├── frequency_encodings.pkl
│       └── model_information.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Notebook_ML_AvecSatisfactionScore.ipynb
│   └── Notebook_ML_SansSatisfactionScore.ipynb
│
│
├── .gitignore
├── README.md
├── requirements.txt


🚀 Application Streamlit


Afin de rendre le modèle de Machine Learning utilisable de manière interactive, deux applications Streamlit ont été développées.

📊 Cas 1 — Avec SatisfactionScore

Cette application utilise le modèle entraîné avec la variable SatisfactionScore.

Elle permet à l'utilisateur de :

saisir les caractéristiques d'un client ;
appliquer automatiquement le prétraitement nécessaire ;
effectuer une prédiction du churn ;
afficher le résultat de la prédiction ;
identifier le risque de départ du client.
📊 Cas 2 — Sans SatisfactionScore

Une deuxième application a été développée sans utiliser la variable SatisfactionScore.

Elle permet également de :

saisir les caractéristiques d'un client ;
appliquer le même processus de prétraitement ;
effectuer une prédiction ;
afficher le résultat obtenu par le modèle.

Cette deuxième version permet notamment de comparer les prédictions avec et sans la variable SatisfactionScore.

🎯 Objectif du déploiement

Le développement de ces deux applications permet de transformer les modèles de Machine Learning en un outil interactif et facilement utilisable, tout en permettant de comparer les deux configurations du projet.

 👩‍💻 Auteur

Asma Ouchen

Projet réalisé dans le cadre de la Licence d'Excellence en Intelligence Artificielle et Analytique des Données.

