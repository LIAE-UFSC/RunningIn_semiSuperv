"""
Single Test Script for Running-In Semi-Supervised Learning

This script demonstrates a complete pipeline for running-in detection in hermetic
alternative compressors using semi-supervised learning. It loads data, trains a
model, and evaluates performance with a confusion matrix visualization.

The script focuses on a single test configuration rather than extensive parameter
sweeps, making it suitable for quick testing and development.

Author: [Your Name]
Date: [Current Date]
"""

# Import required libraries
from utils import RunInSemiSupervised
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if __name__ == "__main__":
    
    # =============================================================================
    # CONFIGURATION PARAMETERS
    # =============================================================================
    
    # Data loading configuration
    compressor_model = "a"  # Use A-series compressor units (A2, A3, A4, A5)
    
    # Feature selection - using only current RMS for this test
    features = ['CorrenteRMS']
    
    # Time series windowing parameters
    sliding_window_size = 5      # Number of time steps in each window
    sliding_window_delay = 3     # Delay between window observations
    movingavg_window_size = 1    # Moving average window size (1 = no smoothing)
    
    # Machine learning configuration
    classifier_type = "LogisticRegression"  # Base classifier for semi-supervised learning
    classifier_params = {}                  # Additional classifier parameters (using defaults)
    
    # Semi-supervised learning parameters
    semisupervised_params = {
        'threshold': 0.6,      # Confidence threshold for pseudo-labeling
        'criterion': 'threshold',  # Criterion for selecting pseudo-labels
        'max_iter': 1000       # Maximum iterations for self-training
    }
    
    # =============================================================================
    # MODEL INITIALIZATION AND TRAINING
    # =============================================================================
    
    # Initialize the semi-supervised learning pipeline
    model = RunInSemiSupervised(
        compressor_model=compressor_model,        # A-series compressor units
        features=features,                         # Current RMS feature only
        window_size=sliding_window_size,          # Time window size
        delay=sliding_window_delay,               # Window delay parameter
        moving_average=movingavg_window_size,     # Moving average smoothing
        balance="undersample",                    # Balance classes via undersampling
        classifier=classifier_type,               # Base classifier type
        semisupervised_params=semisupervised_params,  # Semi-supervised parameters
        classifier_params=classifier_params       # Additional classifier parameters
    )
    
    # Train the model using semi-supervised learning
    print("Training semi-supervised model...")
    model.fit()
    print("Model training completed.")
    
    # =============================================================================
    # MODEL EVALUATION
    # =============================================================================
    
    # Get training data for evaluation
    X_train, y_train = model.get_train_data()
    
    # Generate predictions on training data
    y_pred = model.predict(X_train)
    y_real = y_train
    
    # Filter out unlabeled samples (-1) for evaluation
    # Only evaluate on samples with known labels (0 or 1)
    eval_y_real = y_real[y_real != -1]
    eval_y_pred = y_pred[y_real != -1]
    
    # =============================================================================
    # RESULTS VISUALIZATION
    # =============================================================================
    
    # Generate confusion matrix
    conf_mat = confusion_matrix(eval_y_real, eval_y_pred)
    
    # Create and display confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap='Blues')
    plt.title("Running-In Detection - Confusion Matrix")
    plt.show()
    
    # Print evaluation metrics
    print(f"\nEvaluation Results:")
    print(f"Total samples evaluated: {len(eval_y_real)}")
    print(f"Confusion Matrix:\n{conf_mat}")

# =============================================================================
# LEGACY CONFIGURATION (COMMENTED OUT)
# =============================================================================
# This section contains the original extensive parameter sweep configuration
# that was used for comprehensive testing. It's preserved for reference but
# commented out to focus on single test execution.

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
#     matriz = np.empty((len(N_instantes),len(D_espacamento),len(M_janela,)))
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


