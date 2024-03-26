from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from BaseModelClass import KNN, SVM, LR
import DataUtilizer

N_EST = 500

class RandomForest:
    def crossValidate(X, y):

        grid = {
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        rf = RandomForestClassifier(n_estimators=N_EST, random_state=0)
        utilizer.gridSearch(X, y, rf, grid)
        
    def getOptimal():

        # Manually adjusted by CV results
        model = RandomForestClassifier(n_estimators=N_EST, min_samples_split=2, min_samples_leaf=1,  max_depth=10)
        return model

def customBagging(model, X, y):
    # Create the Bagging ensemble model
    ensemble_model = BaggingClassifier(model, n_estimators=N_EST)
    # Train the ensemble model
    ensemble_model.fit(X, y)
    # Evaluate the ensemble model
    return ensemble_model



utilizer = DataUtilizer.DataUtilizer()

train_filepath = "train.csv"
test_filepath = "test.csv"
X, y = utilizer.getDataFromCSV(train_filepath)
X = utilizer.normalizeData(X)
X = utilizer.pca(X)
test, _ = utilizer.getDataFromCSV(test_filepath)
test = utilizer.pca(test)

model = RandomForest.getOptimal()

for i in range(1, 6):
    ensemble = model.fit(X,y)
    utilizer.getPredictCSV(test, ensemble, "bag_rf_{}.csv".format(i))
