# Importa bibliotecas
from sklearn import semi_supervised
from sklearn import svm
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.cross_decomposition import PLSCanonical
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

''''
classificadores = ['KNN',
                   'RandomForest10',
                   'RandomForest50',
                   'RandomForest100',
                   'RandomForest500',
                   'RandomForest1000',
                   'RandomForest5000',
                   'SVM_Linear',
                   'SVM_Poly2',
                   'SVM_Poly3',
                   'SVM_rbf',
                   'SVM_sigmoid',
                   'Tree'
                   ]
    '''
def modelo_classificador_semi_supervizionado(
        X_func,
        y_func,
        num_clasificador
):
    # -------------------------
    # Classificador supervisionado
    # -------------------------

    if num_clasificador == 0:
        classificador_supervisionado = KNN_Classificador()
    elif num_clasificador == 1:
        classificador_supervisionado = random_Forest(10)
    elif num_clasificador == 2:
        classificador_supervisionado = random_Forest(50)
    elif num_clasificador == 3:
        classificador_supervisionado = random_Forest(100)
    elif num_clasificador == 4:
        classificador_supervisionado = random_Forest(500)
    elif num_clasificador == 5:
        classificador_supervisionado = random_Forest(1000)
    elif num_clasificador == 6:
        classificador_supervisionado = random_Forest(5000)
    elif num_clasificador == 7:
        classificador_supervisionado = SVM('linear')
    elif num_clasificador == 8:
        classificador_supervisionado = SVM_poly(2)
    elif num_clasificador == 9:
        classificador_supervisionado = SVM_poly(3)
    elif num_clasificador == 10:
        classificador_supervisionado = SVM('rbf')
    elif num_clasificador == 11:
        classificador_supervisionado = SVM('sigmoid')
    elif num_clasificador == 12 :
        classificador_supervisionado = Tree()

    lbl = semi_supervised.SelfTrainingClassifier(base_estimator=classificador_supervisionado, threshold=0.8)
    lbl.fit(X_func, y_func)
    y_pred = lbl.predict(X_func)
    return y_pred


def KNN_Classificador():
    return KNeighborsClassifier(n_neighbors=100, weights='distance', algorithm='auto')

def random_Forest(n_estimador):
    return RandomForestClassifier(n_estimators=n_estimador)

def Tree():
    return tree.DecisionTreeClassifier()


def SVM(kernel):
    return svm.SVC(kernel=kernel, probability=True)

def SVM_poly(n):
    return svm.SVC(kernel='poly', probability=True,degree=n)

