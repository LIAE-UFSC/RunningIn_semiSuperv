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

def modelo_classificador_semi_supervizionado(
        X_func,
        y_func,
        classificador
            ):
    
    if classificador[0:3] == 'KNN':
        classificador_supervisionado = KNN_Classificador(int(classificador[3:]))
    elif classificador[0:3] == 'Ran':
        classificador_supervisionado = random_Forest(int(classificador[12:]))
    elif classificador == 'SVM_Linear':
        classificador_supervisionado = SVM('linear')
    elif classificador == 'SVM_Poly2':
        classificador_supervisionado = SVM_poly(2)
    elif classificador == 'SVM_Poly3':
        classificador_supervisionado = SVM_poly(3)
    elif classificador == 'SVM_rbf':
        classificador_supervisionado = SVM('rbf')
    elif classificador == 'SVM_sigmoid':
        classificador_supervisionado = SVM('sigmoid')
    elif classificador == 'Tree' :
        classificador_supervisionado = Tree()
    else:
        raise Exception("Unknown classifier")

    lbl = semi_supervised.SelfTrainingClassifier(base_estimator=classificador_supervisionado, threshold=0.8)
    lbl.fit(X_func, y_func)
    y_pred = lbl.predict(X_func)
    return y_pred


def KNN_Classificador(n=100):
    return KNeighborsClassifier(n_neighbors=n, weights='distance', algorithm='auto',n_jobs=-1)

def random_Forest(n_estimador):
    return RandomForestClassifier(n_estimators=n_estimador,n_jobs=-1)

def Tree():
    return tree.DecisionTreeClassifier()

def SVM(kernel):
    return svm.SVC(kernel=kernel, probability=True)

def SVM_poly(n):
    return svm.SVC(kernel='poly', probability=True,degree=n)

