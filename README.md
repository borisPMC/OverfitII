Dataset evaluation
1. data is well-constructed (100%)
2. limited training set (250 rows)
3. high-dimensionality (300 numeric features)
4. Vast testing set (19750 rows)

Data Cleansing not needed

Data Preprocessing
- Normalization

Candidate models:
- Random Forest (Main)
- kNN (Main)
- Bayesian Tree (Control Test)
- X Deep Learning models

Characteristic for each model training:
Random Forest (Decision Tree based)
- Normalization
kNN
- Normalization
- PCA


Best fit record:
23/3 - knn
Best parameters found: {'weights': 'uniform', 'p': 2, 'n_neighbors': 13}
Best accuracy on validation set: 0.736
Normalized
23/3 - rf
Best parameters found: {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
Best accuracy on validation set: 0.732
Normalized

Submitted record:
23/3 - 1
1. no preprocess
2. RF
3. n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=30, bootstrap=False, random_state=0 
4. Score: 0.5