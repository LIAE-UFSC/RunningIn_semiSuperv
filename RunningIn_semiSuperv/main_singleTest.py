#Import bibliotecas
from utils import RunInSemiSupervised
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay



if __name__ == "__main__":
    # TODO: Implement example of single run of the semi-supervised model

    model = "a"
    
    features = ['CorrenteRMS']

    sliding_window_size = 5
    sliding_window_delay = 3
    movingavg_window_size = 1

    classifier_type = "LogisticRegression"

    classifier_params = {}

    semisupervised_params = {'threshold': 0.6,
                            'criterion': 'threshold',
                            'max_iter': 1000}

    model = RunInSemiSupervised(compressor_model=model,
                                features=features,
                                window_size=sliding_window_size,
                                delay=sliding_window_delay,
                                moving_average= movingavg_window_size,
                                balance="undersample",
                                classifier=classifier_type,
                                semisupervised_params=semisupervised_params,
                                classifier_params=classifier_params)
    
    model.fit()

    X_train, y_train = model.get_train_data()

    y_pred = model.predict(X_train)
    y_real = y_train

    eval_y_real = y_real[y_real != -1]
    eval_y_pred = y_pred[y_real != -1]

    conf_mat = confusion_matrix(eval_y_real, eval_y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Old params:

# #Variaveis globais
# parallel = True
# modelo= 'b'
# classificadores = [
#                     'KNN3',
#                     'KNN6',
#                     'KNN30',
#                     'KNN60',
#                     'KNN300',
#                     'RandomForest10',
#                     'RandomForest100',
#                     'RandomForest1000',
#                     'SVM_Linear',
#                     'SVM_Poly2',
#                     'SVM_Poly3',
#                     'SVM_rbf',
#                     # 'SVM_sigmoid',
#                     # 'Tree'
#                    ]
# lista_grands = [
#         'CorrenteRMS'
#         ]

# N_instantes = [1,2,4,8,16,32]
# D_espacamento = [1,5,10,15,20,25,30]
# M_janela = [1]
# tMin = 1
# tMax= 40
# windowMax = 180

# #
# dataframe_extraido = load_base_data(model= modelo)
# resultados_matriz = open('resultados_matriz','w')
# nome_arquivo_matriz = 'resultados_matriz.txt'
# #


# for classificador in classificadores:
#     matriz = np.empty((len(N_instantes),len(D_espacamento),len(M_janela),))
#     matriz[:] = np.nan
#     criar_pasta(f'resultados\\{classificador}')
    
#     def process_main(i,j,k):
#         if ((N_instantes[i]-1)*D_espacamento[j]) > windowMax:
#             return
        
#         if (((N_instantes[i] == 1) and (D_espacamento[j] > 1))) > windowMax:
#             return
        
#         if os.path.isfile(os.getcwd()+f'\\resultados\\{classificador}\\dados AN{N_instantes[i]}D{D_espacamento[j]}M{M_janela[k]}CLASS{classificador}MODEL{modelo}.png'):
#             return

#         X_modelo, X_dataset, y_pre_categorizado, y = pre_processamento(N_instantes=N_instantes[i],
#                                                                     D_espacamento=D_espacamento[j],
#                                                                     M_janela=M_janela[k],
#                                                                     scaling=True)
#         y_categorizado = modelo_classificador_semi_supervizionado(X_func=X_modelo, y_func=y_pre_categorizado, classificador=classificador)

#         plota_grafico(X_old=X_dataset, func_pred=y_categorizado,
#                     classificador=classificador,
#                     N_instantes=N_instantes[i],
#                     D_espacamento=D_espacamento[j],
#                     M_janela=M_janela[k],
#                     modelo=modelo,
#                     y_old = y
#                         )
        
#         return 1
    
#     if parallel:
#         Parallel(n_jobs=-1, verbose = 10)(delayed(process_main)(i,j,k) for i,j,k in 
#                                         itertools.product(range(len(N_instantes)),
#                                                             range(len(D_espacamento)),
#                                                             range(len(M_janela))))
#     else:
#         for i,j,k in itertools.product(range(len(N_instantes)),
#                     range(len(D_espacamento)),
#                     range(len(M_janela))): process_main(i,j,k)

#     # with open(f"resultados/{classificador}.txt", "w") as arquivo:
#     #     # Percorrer a matriz e escrever cada valor no arquivo
#     #     for camada in matriz:
#     #         for linha in camada:
#     #             for valor in linha:
#     #                 arquivo.write(str(valor) + " ")
#     #             arquivo.write("\n")  # Adicionar uma quebra de linha após cada linha
#     #         arquivo.write("\n")  # Adicionar uma linha em branco entre as camadas


