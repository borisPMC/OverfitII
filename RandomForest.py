import DataUtilizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def crossValidation_RandomForest(X, y):

    random_grid = {
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf = RandomForestClassifier()
    utilizer.randomSearch(X, y, rf, random_grid)
    
def train_RandomForest(X, y):

    # Manually adjusted by CV results
    model = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=2, max_features='sqrt', max_depth=10, bootstrap=True, random_state=0)
    model = model.fit(X, y)
    return model

def getScoreNtier_RandomForest(X, y):
    
    random_grid = {
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
    }

    model = RandomForestClassifier()
    utilizer.plotScoreNTier(X, y, model, random_grid, 10, "Accuracy vs Iteration with RF")


utilizer = DataUtilizer.DataUtilizer()

train_filepath = "train.csv"
test_filepath = "test.csv"

X, y = utilizer.getDataFromCSV(train_filepath)
X = utilizer.normalizeData(X)
# X = utilizer.pca(X)
getScoreNtier_RandomForest(X, y)
# crossValidation_RandomForest(X, y)
# model = train_RandomForest(X, y)

# testCase, empty = utilizer.getDataFromCSV(test_filepath)
# utilizer.getPredictCSV(testCase, model)