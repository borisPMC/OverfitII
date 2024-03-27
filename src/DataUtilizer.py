import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class DataUtilizer:

    @staticmethod
    def getDataFromCSV(path):

        X = pd.DataFrame()
        y = pd.DataFrame()

        try:
            df = pd.read_csv(filepath_or_buffer=path, header=0, index_col=0)
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

        print(X.shape)
        return X, y

    @staticmethod
    def getPredictCSV(X, model, filename="predict.csv"):

        result = pd.DataFrame()
        result["id"] = [i for i in range(250, 20000)]
        result.set_index("id", inplace=True)
        result["target"] = model.predict(X)
        result.to_csv(filename)
        

    @staticmethod
    def normalizeData(X):
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return df_normalized
    
    @staticmethod
    def pca(X):
        pca = PCA(n_components=60, random_state=0) # 300 * 0.2 = 60
        result = pca.fit_transform(X)
        return result

    
    @staticmethod
    def gridSearch(X, y, model, gridDict):

        # Perform Randomized Search Cross Validation
        gridModel = GridSearchCV(model, param_grid=gridDict, cv = 5, scoring='accuracy')
        gridModel.fit(X, y)

        # Evaluate the best model
        best_params = gridModel.best_params_
        best_score = gridModel.best_score_

        print("Best parameters found:", best_params)
        print("Best accuracy on validation set:", best_score)

        return gridModel.cv_results_
    
    @staticmethod
    def changeKeyName(grid: dict):
        for s in list(grid):
            grid["estimator__"+s] = grid[s]
            del grid[s]

        return grid