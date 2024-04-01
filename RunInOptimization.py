import pandas as pd
import numpy as np
from buildDataset import separa_xy_runin, preproc_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from skopt import gp_minimize
from imblearn.under_sampling import RandomUnderSampler

def runningin_val_score(estimator, data:pd.DataFrame, scorer) -> float:
    """Avalia a validação cruzada por ensaios de amaciamento

    Parâmetros
    ----------
    estimator : estimador
        Modelo estimador para aplicar "fit"

    data: pd.DataFrame
        Dados do conjunto

    scorer: callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.

    Retorna
    -------
    score: float
        Média do score dos modelos em ensaios de amaciamento

    modelAll: 
        Modelo treinado com todos os ensaios de treino
    """

    # Seleciona as unidades com ensaios de amaciamento
    unidades = np.unique(data.loc[data["N_ensaio"]==0,"Unidade"].tolist())

    score =[]
    model = []
    for k,un in enumerate(unidades):

        # Separa o ensaio de amaciamento de uma unidade p/ teste e os ensaios de outras unidades p/ treino
        train = data[data["Unidade"]!=un]
        test = data[(data["N_ensaio"]==0)&(data["Unidade"]==un)]

        X_train, y_train = separa_xy_runin(train)
        X_test, y_test = separa_xy_runin(test)

        # Treina o modelo e aplica o score
        model.append(estimator.fit(X_train,y_train))
        score.append(scorer(model[k],X_test,y_test))

    X,y = separa_xy_runin(data)

    # Modelo com todos os ensaios de treino
    modelAll = estimator.fit(X,y)

    return score, modelAll

def optimizeModel(DataTrain:pd.DataFrame,grandezas: list[str], modelo: str, search_space, scoring, tMax: float, tMin:float = 1, dropTempo:bool = True):
    """Otimização de modelos p/ detecção de amaciamento

    Parâmetros
    ----------
    DataTrain : pd.Dataframe
        Dados do conjunto

    grandeza: list[str]
        Lista com o nome das grandezas desejadas no formato "Nicolas". Nomes disponíveis:
                    'CorrenteRMS'               'CorrenteVariancia'            'CorrenteCurtose'
            'VibracaoCalotaInferiorRMS' 'VibracaoCalotaInferiorCurtose' 'VibracaoCalotaInferiorVariancia'
            'VibracaoCalotaSuperiorRMS' 'VibracaoCalotaSuperiorCurtose' 'VibracaoCalotaSuperiorVariancia'
                        'Vazao'

    modelo: str
        Modelo de ML para otimização. Modelos disponíveis:
            "RF": random forest
            "SVM_Lin": SVM de kernel linear
            "SVM_Pol": SVM de kernel polinomial
            "SVM_Gauss": SVM de kernel gaussiano
            "kNN": k-nearest neighbors
            "LogReg": regressão logística

    search_space: [list, shape (n_dims,)
        Lista de dimensões da busca. Sendo inf e sup os limites de cada dimensão, o espaço deve ser definido da seguinte forma:
        search_space = [
            # Para todos os modelos:
            Integer(inf,sup, name= 'N_instantes'), # Número de pontos na janela de sliding window
            Integer(inf,sup, name= 'D_espacamento'), # Atraso entre amostras p/ sliding window
            Integer(inf,sup, name= 'M_janela'), # Largura do filtro de médias móveis

            # Caso modelo = "RF":
            Integer(inf,sup, name= 'max_depth'), # Profundidade máxima das árvores
            Integer(inf,sup,name= 'min_samples_leaf') # Número mínimo de amostras por folha

            # Caso modelo = "SVM_Lin" ou "SVM_Gauss"
            Real(inf,sup, name = 'C') # Parâmetro C de regularização L2

            # Caso modelo = "SVM_Pol"
            Real(inf,sup, name = 'C') # Parâmetro C de regularização L2
            Integer(inf,sup, name = 'PolDeg') # Grau do polinômio

            # Caso modelo = "kNN"
            Integer(inf,sup, name = 'kNN') # Número K de vizinhos
            ]


    scoring: callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.
    
    tMax: float
        Instante máximo para análise

    tMin: float
        Instante mínimo para análise (default = 1)

    dropTempo: bool
        Manter ou não coluna do tempo no dataset gerado (default = True)

    Retorna
    -------
    minModel:
        Resultado ótimo dentro do espaço de busca
    """


    def _train_and_score(parameter) -> float:
        N_instantes = parameter[0]
        D_espacamento = parameter[1]
        M_janela = parameter[2]

        _, dataset, _ = preproc_dataset(DataTrain, grandezas, M_janela, N_instantes, D_espacamento, tempoMin = tMin, tempoMax = tMax, dropTempo = dropTempo, scale=True)
        X, y = RandomUnderSampler(sampling_strategy='majority', random_state= 42).fit_resample(dataset.drop(["Amaciado"],axis=1), dataset["Amaciado"])
        dataset = pd.concat([X,y],axis=1)

        if modelo == "RF":
            max_depth = parameter[3]
            min_samples_leaf = parameter[4]
            model = RandomForestClassifier(random_state=42, n_jobs=-1, bootstrap=True, max_depth = max_depth, min_samples_leaf = min_samples_leaf, n_estimators = 50)
        elif modelo == "SVM_Lin":
            c_regular = parameter[3]
            model = SVC(random_state=42, kernel = 'linear', C = c_regular)
        elif modelo == "SVM_Pol":
            c_regular = parameter[3]
            pol_deg = parameter[4]
            model = SVC(random_state=42, kernel = 'poly',coef0=pol_deg, C = c_regular)
        elif modelo == "SVM_Gauss":
            c_regular = parameter[3]
            model = SVC(random_state=42, kernel = 'rbf', C = c_regular)
        elif modelo == "kNN":
            knn = parameter[3]
            model = KNeighborsClassifier(n_jobs=-1, n_neighbors=knn)
        elif modelo == "LogReg":
            model = LogisticRegression(random_state=42, n_jobs=-1, penalty = 'none', max_iter=1000)
        else:
            raise ValueError("Model not accepted")

        score, _ = runningin_val_score(model,dataset,scoring)

        return 1-np.mean(score)
    if modelo == "SVM_Pol":
        minModel = gp_minimize(_train_and_score,search_space,n_initial_points= 30, n_jobs= -1, n_calls= 75,random_state= 42)
    else:
        minModel = gp_minimize(_train_and_score,search_space,n_initial_points= 75, n_jobs= -1, n_calls= 150,random_state= 42)

    return minModel

def train_and_score(parameter, DataTrain:pd.DataFrame, DataTest:pd.DataFrame,grandezas: list[str], modelo: str, scoring, tMax: float, tMin:float = 1, dropTempo:bool = True):
    """Treinamento e score para modelos de amaciamento

    Parâmetros
    ----------
    parameter: list[Any]
        Parâmetros de processamento e modelo dados pela função optimizeModel()

    DataTrain : pd.Dataframe
        Dados do conjunto para treino

    DataTrain : pd.Dataframe
        Dados do conjunto para teste

    grandeza: list[str]
        Lista com o nome das grandezas desejadas no formato "Nicolas". Nomes disponíveis:
                    'CorrenteRMS'               'CorrenteVariancia'            'CorrenteCurtose'
            'VibracaoCalotaInferiorRMS' 'VibracaoCalotaInferiorCurtose' 'VibracaoCalotaInferiorVariancia'
            'VibracaoCalotaSuperiorRMS' 'VibracaoCalotaSuperiorCurtose' 'VibracaoCalotaSuperiorVariancia'
                        'Vazao'

    modelo: str
        Modelo de ML para otimização. Modelos disponíveis:
            "RF": random forest
            "SVM_Lin": SVM de kernel linear
            "SVM_Pol": SVM de kernel polinomial
            "SVM_Gauss": SVM de kernel gaussiano
            "kNN": k-nearest neighbors
            "LogReg": regressão logística

    scoring: callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.
    
    tMax: float
        Instante máximo para análise

    tMin: float
        Instante mínimo para análise (default = 1)

    dropTempo: bool
        Manter ou não coluna do tempo no dataset gerado (default = True)

    Retorna
    -------
    modelTrained: modelo
        Escalonador e modelo treinados

    scoreTrain: float
        Média do score dos modelos em ensaios de amaciamento de treino

    scoreTest: float
        Média do score dos dados de teste
    """

    N_instantes = parameter[0]
    D_espacamento = parameter[1]
    M_janela = parameter[2]

    _, datasetTrain, scaler = preproc_dataset(DataTrain, grandezas, M_janela, N_instantes, D_espacamento, tempoMin = tMin, tempoMax = tMax, dropTempo = dropTempo, scale=True)
    X, y = RandomUnderSampler(sampling_strategy='majority', random_state= 42).fit_resample(datasetTrain.drop(["Amaciado"],axis=1), datasetTrain["Amaciado"])
    datasetTrain = pd.concat([X,y],axis=1)

    if modelo == "RF":
        max_depth = parameter[3]
        min_samples_leaf = parameter[4]
        model = RandomForestClassifier(random_state=42, n_jobs=-1, bootstrap=True, max_depth = max_depth, min_samples_leaf = min_samples_leaf, n_estimators = 50)
    elif modelo == "SVM_Lin":
        c_regular = parameter[3]
        model = SVC(random_state=42, kernel = 'linear', C = c_regular)
    elif modelo == "SVM_Pol":
        c_regular = parameter[3]
        pol_deg = parameter[4]
        model = SVC(random_state=42, kernel = 'poly',coef0=pol_deg, C = c_regular)
    elif modelo == "SVM_Gauss":
        c_regular = parameter[3]
        model = SVC(random_state=42, kernel = 'rbf', C = c_regular)
    elif modelo == "kNN":
        knn = parameter[3]
        model = KNeighborsClassifier(n_jobs=-1, n_neighbors=knn)
    elif modelo == "LogReg":
        model = LogisticRegression(random_state=42, n_jobs=-1, penalty = 'none')
    else:
        raise ValueError("Model not accepted")

    scoreTrain, modelTrained = runningin_val_score(model,datasetTrain,scoring)

    datasetTest = preproc_dataset(DataTest, grandezas, M_janela, N_instantes, D_espacamento, tempoMin = tMin, tempoMax = tMax, dropTempo = dropTempo, scale=False)
    XTest, yTest = separa_xy_runin(datasetTest)

    scoreTest = scoring(modelTrained,scaler.transform(XTest),yTest)

    modelTrained = Pipeline(
                        [
                            ("scaler",scaler),
                            ("model",modelTrained)
                        ]
    )

    return modelTrained, np.mean(scoreTrain), scoreTest
