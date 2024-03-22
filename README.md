Dataset evaluation
1. data is well-constructed (100%)
2. limited training set (250 rows)
3. high-dimensionality (301 numeric features)

Data Cleansing not needed

Data Preprocessing not needed (unknown features)

Candidate models:
- Random Forest (Main)
- kNN (Control Test)
- Decision Tree (Control Test)
- Bayesian Tree (Control Test)
- X Deep Learning models

Best fit record:
22/3 - RandomForestClassifier - {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False} - 0.736