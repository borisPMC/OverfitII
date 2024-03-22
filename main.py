import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def getDataFromCSV(path):

    X = pd.DataFrame()
    y = pd.DataFrame()

    try:
        df = pd.read_csv(filepath_or_buffer=path)
    except:
        print("CSV file not found at {}, please revise!".format(path))
        return (X, y)

    if "target" in df.columns:
        y = df["target"]
        X = df.drop("target", axis=1)
        print("Train data imported")
    else:
        X = df
        print("Test data imported")

    return X, y

def crossValidation_RandomForest(X, y):

    # Number of trees in random forest
    n_estimators = [i*200 for i in range(1, 11)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [i*10 for i in range(1, 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    gridModel = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    gridModel = gridModel.fit(X,y)

    print(gridModel.best_params_)
    print("Best Accuracy: {}".format(gridModel.best_score_))
    
def train_RandomForest(X, y):

    # Manually adjusted by CV results

    model = RandomForestClassifier(n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=70, bootstrap=False)
    model = model.fit(X, y)
    return model

def getPredictCSV()

train_filepath = "J:\\COMProfile\\Documents\\GitHub\\OverfitII\\train.csv"
test_filepath = "J:\\COMProfile\\Documents\\GitHub\\OverfitII\\test.csv"

X, y = getDataFromCSV(train_filepath)
model = train_RandomForest(X, y)

testCase = getDataFromCSV(test_filepath)




