#Import bibliotecas
import numpy as np

from analise_modelo import analise_modelos
from pre_processador import retira_y,prep_data,label_df,separa_Xy,retira_valores
from modelos import modelo_classificador_semi_supervizionado
from sklearn.metrics import matthews_corrcoef
from plotagem import plota_grafico
import os
#Variaveis globais

modelo= 'a'
classificadores = [# 'KNN',
                #    'RandomForest10',
                #    'RandomForest50',
                #    'RandomForest100',
                #    'RandomForest500',
                #    'RandomForest1000',
                #    'RandomForest5000',
                   'SVM_Linear',
                   'SVM_Poly2',
                   'SVM_Poly3',
                   'SVM_rbf',
                   'SVM_sigmoid',
                   'Tree'
                   ]
lista_grands = [
        'CorrenteRMS'
        ]

N_instantes = [1,2,4,8,16,32]
D_espacamento = [1,5,10,15,20,25,30]
M_janela = [1]
tMin = 1
tMax= 20
windowMax = 180

#
dataframe_extraido = analise_modelos(modelo= modelo)
resultados_matriz = open('resultados_matriz','w')
nome_arquivo_matriz = 'resultados_matriz.txt'
#
def pre_processamento(N_instantes,D_espacamento,M_janela):
    dataframe_preparado = prep_data(N_instantes=N_instantes,D_espacamento=D_espacamento,M_janela=M_janela,df_analise=dataframe_extraido,lista_grands=lista_grands,tMin=tMin,tMax=tMax)
    dataframe_categorizado = label_df(dataframe_preparado)
    X_dataset, y = separa_Xy(dataframe_categorizado)
    X_modelo = retira_valores(X_dataset)
    y_pre_categorizado = retira_y(dataframe_categorizado)
    return X_modelo,X_dataset,y_pre_categorizado, y

def criar_pasta(caminho):
    try:
        os.makedirs(caminho)
        print(f"Pasta '{caminho}' criada com sucesso.")
    except FileExistsError:
        print(f"A pasta '{caminho}' já existe.")


for x in range(len(classificadores)):
    matriz = np.empty((len(N_instantes),len(D_espacamento),len(M_janela),))
    matriz[:] = np.nan;
    criar_pasta(f'resultados\\{classificadores[x]}')
    for i in range(len(N_instantes)):
        for j in range(len(D_espacamento)):
            for k in range(len(M_janela)):
                if ((N_instantes[i]-1)*D_espacamento[j]) > windowMax:
                    break

                X_modelo, X_dataset, y_pre_categorizado, y = pre_processamento(N_instantes=N_instantes[i],
                                                                               D_espacamento=D_espacamento[j],
                                                                               M_janela=M_janela[k])
                y_categorizado = modelo_classificador_semi_supervizionado(X_func=X_modelo, y_func=y_pre_categorizado, num_clasificador=x)

                plota_grafico(X_old=X_dataset, func_pred=y_categorizado,
                              classificador=classificadores[x],
                              N_instantes=N_instantes[i],
                              D_espacamento=D_espacamento[j],
                              M_janela=M_janela[k],
                              modelo=modelo,
                              y_old = y
                                )

                coeficiente_matthews = matthews_corrcoef(y[y!=-1], y_categorizado[y!=-1])
                matriz[i][j][k] = coeficiente_matthews

    with open(f"resultados/{classificadores[x]}.txt", "w") as arquivo:
        # Percorrer a matriz e escrever cada valor no arquivo
        for camada in matriz:
            for linha in camada:
                for valor in linha:
                    arquivo.write(str(valor) + " ")
                arquivo.write("\n")  # Adicionar uma quebra de linha após cada linha
            arquivo.write("\n")  # Adicionar uma linha em branco entre as camadas

