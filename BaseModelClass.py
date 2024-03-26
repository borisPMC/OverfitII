import DataUtilizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class KNN:
    def plotCvResult(results):
        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.scatter(results['param_n_neighbors'], results['mean_test_score'], marker='o')
        plt.title('Accuracy vs Number of Neighbors')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Test Accuracy')
        plt.grid(True)
        plt.savefig("result.png")

    def crossValidate(X, y):
        param_grid = {
        'n_neighbors': [i for i in range(1, 32, 2)],
        }
        model = KNeighborsClassifier()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # plotCvResult(gridResult)

    def getOptimal():

        model = KNeighborsClassifier(n_neighbors=17)
        return model
        
class SVM:
    def plotCvResult(results, param_grid):

        # Plot graphs for each hyperparameter
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot accuracy vs C
        C_values = np.unique(param_grid['C'])
        C_accuracy = [np.max(results['mean_test_score'][results['param_C'] == C_value]) for C_value in C_values]
        axs[0, 0].plot(C_values, C_accuracy, marker='o')
        axs[0, 0].set_title('Accuracy vs C')
        axs[0, 0].set_xlabel('C')
        axs[0, 0].set_ylabel('Max Test Accuracy')
        axs[0, 0].set_xscale('log')

        # Plot accuracy vs kernel
        kernel_values = np.unique(param_grid['kernel'])
        kernel_accuracy = [np.max(results['mean_test_score'][results['param_kernel'] == kernel_value]) for kernel_value in kernel_values]
        axs[0, 1].bar(kernel_values, kernel_accuracy)
        axs[0, 1].set_title('Accuracy vs Kernel')
        axs[0, 1].set_xlabel('Kernel')
        axs[0, 1].set_ylabel('Max Test Accuracy')

        # Plot accuracy vs gamma
        gamma_values = np.unique(param_grid['gamma'])
        gamma_accuracy = [np.max(results['mean_test_score'][results['param_gamma'] == gamma_value]) for gamma_value in gamma_values]
        axs[1, 0].bar(gamma_values, gamma_accuracy)
        axs[1, 0].set_title('Accuracy vs Gamma')
        axs[1, 0].set_xlabel('Gamma')
        axs[1, 0].set_ylabel('Max Test Accuracy')

        # Remove empty subplot
        fig.delaxes(axs[1, 1])

        plt.tight_layout()
        plt.savefig("result.png")


    def crossValidate(X, y):
        param_grid = {
            'C': [0.1, 1, 10, 100],              # Regularization parameter
            'kernel': ['linear', 'rbf', 'poly'], # Kernel type
            'gamma': ['scale', 'auto'],           # Kernel coefficient for 'rbf' and 'poly' kernels
        }
        model = SVC()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # plotCvResult(gridResult, param_grid)

    def getOptimal():
        
        model = SVC(C=0.1, gamma="scale", kernel="rbf")
        return model
    
class LR:
    def plotCvResult(results, param_grid):
        # Plot graphs for each hyperparameter
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot accuracy vs penalty
        penalty_values = np.unique(param_grid['penalty'])
        penalty_accuracy = [np.max(results['mean_test_score'][results['param_penalty'] == penalty_value]) for penalty_value in penalty_values]
        axs[0, 0].bar(penalty_values, penalty_accuracy)
        axs[0, 0].set_title('Accuracy vs Penalty')
        axs[0, 0].set_xlabel('Penalty')
        axs[0, 0].set_ylabel('Max Test Accuracy')

        # Plot accuracy vs C
        C_values = np.unique(param_grid['C'])
        C_accuracy = [np.max(results['mean_test_score'][results['param_C'] == C_value]) for C_value in C_values]
        axs[0, 0].plot(C_values, C_accuracy, marker='o')
        axs[0, 1].set_title('Accuracy vs C')
        axs[0, 1].set_xlabel('C')
        axs[0, 1].set_ylabel('Max Test Accuracy')
        axs[0, 1].set_xscale('log')

        # Plot accuracy vs solver
        solver_values = np.unique(results['param_solver'])
        solver_accuracy = [np.max(results['mean_test_score'][results['param_solver'] == solver_value]) for solver_value in solver_values]
        axs[1, 0].bar(solver_values, solver_accuracy)
        axs[1, 0].set_title('Accuracy vs Solver')
        axs[1, 0].set_xlabel('Solver')
        axs[1, 0].set_ylabel('Max Test Accuracy')

        # Remove empty subplot
        fig.delaxes(axs[1, 1])

        plt.tight_layout()
        plt.savefig("result.png")

    def crossValidate(X, y):
        param_grid = {
        'penalty': ['l1', 'l2'],                  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10, 100],      # Inverse of regularization strength
        'solver': ['liblinear', 'saga'],          # Algorithm to use in the optimization problem
        }
        model = LogisticRegression()
        utilizer = DataUtilizer.DataUtilizer()
        gridResult = utilizer.gridSearch(X, y, model, param_grid)

        # LR.plotCvResult(gridResult, param_grid)

    def getOptimal():
        
        model = LogisticRegression(C=0.1, penalty="l1", solver="liblinear")
        return model