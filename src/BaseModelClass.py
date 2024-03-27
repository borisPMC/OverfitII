import DataUtilizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression

class KNN:
    def crossValidate(X, y):
        param_grid = KNN.getParamGrid()
        model = KNeighborsClassifier()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # plotCvResult(gridResult)

    def getOptimal():
        model = KNeighborsClassifier(n_neighbors=13)
        return model
    
    def getBagOptimal():
        model = KNeighborsClassifier(n_neighbors=9)

    def getRaw():
        return KNeighborsClassifier()
    
    def getParamGrid():
        grid = {
            'n_neighbors': [i for i in range(1, 32, 2)],
        }

        return grid
        
class SVM:
    def crossValidate(X, y):
        param_grid = SVM.getParamGrid()
        model = SVC()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # plotCvResult(gridResult, param_grid)

    def getOptimal():
        model = SVC(C=10, gamma="scale", kernel="rbf")
        return model
    
    def getBagOptimal():
        model = SVC(C=0.1, gamma="scale", kernel="linear")
        return model

    def getAdaOptimal():
        model = SVC(C=1, gamma="scale", kernel="linear")
        return model

    def getRaw():
        return SVC()
    
    def getParamGrid():
        grid = {
            'C': [0.1, 1, 10, 100],              # Regularization parameter
            'kernel': ['linear', 'rbf', 'poly'], # Kernel type
            'gamma': ['scale', 'auto'],           # Kernel coefficient for 'rbf' and 'poly' kernels
        }
        return grid
    
class LogR:
    def crossValidate(X, y):
        param_grid = LogR.getParamGrid()
        model = LogisticRegression()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # LR.plotCvResult(gridResult, param_grid)

    def getOptimal():
        model = LogisticRegression(C=0.1, penalty="l1", solver="liblinear")
        return model
    
    def getBagOptimal():
        model = LogisticRegression(C=0.1, penalty="l1", solver="liblinear")
        return model

    def getStackOptimal():
        model = LogisticRegression(C=100)
        return model

    def getRaw():
        return LogisticRegression()
    
    def getParamGrid():
        grid = {
            'penalty': ['l1', 'l2'],                  # Regularization penalty
            'C': [0.001, 0.01, 0.1, 1, 10, 100],      # Inverse of regularization strength
            'solver': ['liblinear', 'saga'],          # Algorithm to use in the optimization problem
        }
        return grid