# Importa bibliotecas
import matplotlib.pyplot as plt
from filtros import filtro

def plota_grafico(X_old,func_pred,
                  classificador,
                  N_instantes,
                  D_espacamento,
                  M_janela,
                  modelo, y_old = None):
    def cor_plot2(array):
        df = ["blue" if x == -1 else "black" if x != 1 else "gray" for x in array]

        return df

    def cor_plot(array):
        # df = pd.DataFrame(index=range(len(y)),columns=range(1) )
        df = ["black" if x != 1 else "red" for x in array]

        return df

    if y_old is None:
        dfSave = X_old[["Tempo","N_ensaio","Unidade"]].assign(runinEst = func_pred)
    else:
        dfSave = X_old[["Tempo","N_ensaio","Unidade"]].assign(runinEst = func_pred, runinOg = y_old)
    dfSave.to_csv(f'resultados/{classificador}/dados AN{N_instantes}D{D_espacamento}M{M_janela}CLASS{classificador}MODEL{modelo}.csv')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Running-in Complete', markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Running-in Incomplete', markerfacecolor='black', markersize=10)
    ]

    if modelo == 'a':
        X_final, y_final, X_a2, X_a3, X_a4, X_a5, y_a2, y_a3, y_a4, y_a5, Xy = filtro(func_pred, X_old,modelo)

            ################
            # Graficos A
            ################

            # plot 1:
        cor_a2 = cor_plot(y_a2)
        plt.subplot(2, 2, 1)
        plt.scatter(X_a2['Tempo'], X_a2[f'CorrenteRMS_MA_{M_janela}'], c=cor_a2, alpha=0.5)
        plt.title('Unidade A1')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')

            # plot 2:
        cor_a3 = cor_plot(y_a3)
        plt.subplot(2, 2, 2)
        plt.scatter(X_a3['Tempo'], X_a3[f'CorrenteRMS_MA_{M_janela}'], c=cor_a3, alpha=0.5)
        plt.title('Unidade A2')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')

            # plot 3:
        cor_a4 = cor_plot(y_a4)
        plt.subplot(2, 2, 3)
        plt.scatter(X_a4['Tempo'], X_a4[f'CorrenteRMS_MA_{M_janela}'], c=cor_a4, alpha=0.5)
        plt.title('Unidade A3')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

            # plot 4:
        cor_a5 = cor_plot(y_a5)
        plt.subplot(2, 2, 4)
        plt.scatter(X_a5['Tempo'], X_a5[f'CorrenteRMS_MA_{M_janela}'], c=cor_a5, alpha=0.5)
        plt.title('Unidade A4')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

    else:
        X, y, X_b5, X_b7, X_b8, X_b10, X_b11, X_b12, X_b15, y_b5, y_b7, y_b8, y_b10, y_b11, y_b12, y_b15, Xy = filtro(func_pred, X_old,modelo)            

        # Plot for b5
        cor_b5 = cor_plot(y_b5)
        plt.subplot(2, 4, 1)
        plt.scatter(X_b5['Tempo'], X_b5[f'CorrenteRMS_MA_{M_janela}'], c=cor_b5, alpha=0.5)
        plt.title('Unidade B5')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')

        # Plot for b7
        cor_b7 = cor_plot(y_b7)
        plt.subplot(2, 4, 2)
        plt.scatter(X_b7['Tempo'], X_b7[f'CorrenteRMS_MA_{M_janela}'], c=cor_b7, alpha=0.5)
        plt.title('Unidade B7')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')

        # Plot for b8
        cor_b8 = cor_plot(y_b8)
        plt.subplot(2, 4, 3)
        plt.scatter(X_b8['Tempo'], X_b8[f'CorrenteRMS_MA_{M_janela}'], c=cor_b8, alpha=0.5)
        plt.title('Unidade B8')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

        # Plot for b10
        cor_b10 = cor_plot(y_b10)
        plt.subplot(2, 4, 4)
        plt.scatter(X_b10['Tempo'], X_b10[f'CorrenteRMS_MA_{M_janela}'], c=cor_b10, alpha=0.5)
        plt.title('Unidade B10')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

        # Plot for b11
        cor_b11 = cor_plot(y_b11)
        plt.subplot(2, 4, 5)
        plt.scatter(X_b11['Tempo'], X_b11[f'CorrenteRMS_MA_{M_janela}'], c=cor_b11, alpha=0.5)
        plt.title('Unidade B11')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

        # Plot for b12
        cor_b12 = cor_plot(y_b12)
        plt.subplot(2, 4, 6)
        plt.scatter(X_b12['Tempo'], X_b12[f'CorrenteRMS_MA_{M_janela}'], c=cor_b12, alpha=0.5)
        plt.title('Unidade B12')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')

        # Plot for b15
        cor_b15 = cor_plot(y_b15)
        plt.subplot(2, 4, 7)
        plt.scatter(X_b15['Tempo'], X_b15[f'CorrenteRMS_MA_{M_janela}'], c=cor_b15, alpha=0.5)
        plt.title('Unidade B15')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.ylabel('Corrente [A]')
        plt.xlabel('Tempo [h]')
    plt.savefig(f'resultados/{classificador}/dados AN{N_instantes}D{D_espacamento}M{M_janela}CLASS{classificador}MODEL{modelo}')
