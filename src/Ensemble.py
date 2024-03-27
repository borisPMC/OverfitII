from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
import xgboost as xgb
from BaseModelClass import KNN, SVM, LogR
import DataUtilizer

class RandomForest:

    def crossValidate(X, y):
        grid = RandomForest.getParamGrid()
        rf = RandomForestClassifier()
        utilizer.gridSearch(X, y, rf, grid)

    def getOptimal():
        # Manually adjusted by CV results
        model = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_depth=None, max_features='sqrt', max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)
        return model
    
    def getParamGrid():
        param_grid = {
            'n_estimators': [10, 50, 100, 200],         # Number of trees in the forest
            'max_depth': [None, 10, 20, 30],            # Maximum depth of the trees
            'max_samples': [0.5, 0.7, 1.0],             # Proportion of samples to draw for training each base estimator
            'max_features': [0.5, 0.7, 1.0],            # Proportion of features to consider when looking for the best split
            'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required at each leaf node
            'max_features': ['auto', 'sqrt'],           # Number of features to consider at each split
            'bootstrap': [True, False],                 # Whether bootstrap samples are used when building trees
            'random_state': [42],                       # Random seed for reproducibility
            'n_jobs': [-1]
        }
        return param_grid

class CustomBagging:

    def crossValidate(X, y, base, base_grid):
        grid = CustomBagging.getParamGrid(base, base_grid)
        model = BaggingClassifier()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, grid)
    
    def getParamGrid(estimator, base_grid):
        result = {}
        grid = {
            'n_estimators': [10, 50, 100],  # Number of base estimators
            'max_samples': [0.5, 0.7, 1.0],  # Proportion of samples to draw for training each base estimator
            'max_features': [0.5, 0.7, 1.0],  # Proportion of features to consider when looking for the best split
            'bootstrap': [True, False],  # Whether samples are drawn with replacement (bootstrap) or without replacement
            'estimator': [estimator],  # Base estimator to be used
            'random_state': [42],  # Random seed for reproducibility
            'n_jobs': [-1]
        }
        base_grid = utilizer.changeKeyName(base_grid)
        result.update(grid)
        result.update(base_grid)
        return result

    def getOptimal(base): # input: model type
        match base:
            case "KNN":
                model = KNN.getBagOptimal()
                ensemble = BaggingClassifier(model, n_estimators=10, max_features=0.5, max_samples=1.0, n_jobs=-1, random_state=42)
            case "SVM":
                model = SVM.getBagOptimal()
                ensemble = BaggingClassifier(model, n_estimators=10, max_features=0.7, max_samples=1.0, n_jobs=-1, random_state=42)
            case "LogR":
                model = LogR.getBagOptimal()
                ensemble = BaggingClassifier(model, n_estimators=100, max_features=1.0, max_samples=0.7, n_jobs=-1, random_state=42)
            case _:
                print("Wrong Augment!")
        return ensemble

class AdaBoosting:

    def crossValidate(X, y, base, base_grid):
        grid = AdaBoosting.getParamGrid(base, base_grid)
        model = AdaBoostClassifier()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, grid)
    
    def getParamGrid(estimator, base_grid):
        result = {}
        grid = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 1],
            'algorithm': ['SAMME', 'SAMME.R'],
            'estimator': [estimator],
            'random_state': [42]
        }
        base_grid = utilizer.changeKeyName(base_grid)
        result.update(grid)
        result.update(base_grid)
        return result

    def getOptimal(base): # input: model type
        match base:
            case "SVM":
                model = SVM.getAdaOptimal()
                ensemble = AdaBoostClassifier(model, learning_rate=1, n_estimators=50, algorithm="SAMME", random_state=42)
            case "LogR":
                model = LogR.getOptimal()
                ensemble = AdaBoostClassifier(model, learning_rate=1, n_estimators=10, algorithm="SAMME", random_state=42)
            case _:
                print("Wrong Augment!")
        return ensemble

class xgBoosting:

    def crossValidate(X, y):
        grid = xgBoosting.getParamGrid()
        model = xgb.XGBClassifier()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, grid)
    
    def getParamGrid():
        param_grid = {
            'n_estimators': [50, 100, 200],      # Number of boosting rounds
            'learning_rate': [0.01, 0.1, 1],     # Step size shrinkage used in each boosting round
            'max_depth': [3, 5, 7],              # Maximum depth of each tree
            'subsample': [0.6, 0.8, 1.0],        # Subsample ratio of the training instances
            'colsample_bytree': [0.6, 0.8, 1.0], # Subsample ratio of columns when constructing each tree
            'gamma': [0, 0.1, 0.2],              # Minimum loss reduction required to make a further partition on a leaf node
            'reg_alpha': [0, 0.1, 0.5],          # L1 regularization term on weights
            'reg_lambda': [0, 0.1, 0.5]          # L2 regularization term on weights
        }
        return param_grid
    
    def getOptimal():
        model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, subsample=0.6, colsample_bytree=0.6, gamma=0.2, reg_alpha=0.5, reg_lambda=0.5)
        return model
    
class stackingModel:

    def crossValidate(X, y, meta):
        grid = stackingModel.getParamGrid()
        base = stackingModel.getBaseModelGrid()
        model = StackingClassifier(base, meta, cv=5, n_jobs=-1)
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, grid)

    def getOptimal(meta_model):
        grid = stackingModel.getBaseModelGrid()
        model = StackingClassifier(grid, meta_model, cv=5, n_jobs=-1)
        return model

    def getParamGrid():
        param_grid = {
            'final_estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],      # Inverse of regularization strength
            'stack_method': ['auto', 'predict_proba', 'decision_function'],
        }
        return param_grid

    def getBaseModelGrid():
        base_models = [
            ('knn', KNN.getOptimal()),
            ('svm', SVM.getOptimal()),
            ('lr', LogR.getOptimal()),
            ('bag_knn', CustomBagging.getOptimal("KNN")),
            ('bag_svm', CustomBagging.getOptimal("KNN")),
            ('ada_svm', AdaBoosting.getOptimal("SVM")),
            ('xgb', xgBoosting.getOptimal())
        ]
        return base_models

utilizer = DataUtilizer.DataUtilizer()

train_filepath = "train.csv"
test_filepath = "test.csv"
X, y = utilizer.getDataFromCSV(train_filepath)
X = utilizer.normalizeData(X)
test, _ = utilizer.getDataFromCSV(test_filepath)

meta = LogR.getStackOptimal()
model = stackingModel.getOptimal(meta)
model.fit(X, y)
utilizer.getPredictCSV(test, model, "stack_2.csv")
