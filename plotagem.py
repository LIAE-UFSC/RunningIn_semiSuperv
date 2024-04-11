# Importa bibliotecas
import matplotlib.pyplot as plt
from filtros import filtro_A

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

    X_final, y_final, X_a2, X_a3, X_a4, X_a5, y_a2, y_a3, y_a4, y_a5, Xy = filtro_A(func_pred, X_old)

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

    plt.savefig(f'resultados/{classificador}/dados AN{N_instantes}D{D_espacamento}M{M_janela}CLASS{classificador}MODEL{modelo}')
