import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class Estatistica:
    def __init__(self, n_amostras=1000, tamanho_amostra=30):
        self.n_amostras = n_amostras
        self.tamanho_amostra = tamanho_amostra
        self.dados = np.random.normal(loc=0, scale=1, size=(n_amostras, tamanho_amostra))

    def teorema_central_limite(self):
        medias = self.dados.mean(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(medias, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribuição das Médias (Teorema Central do Limite)', fontsize=16)
        ax.set_xlabel('Média das Amostras', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        
        media_amostral = np.mean(medias)
        ax.axvline(media_amostral, color='red', linestyle='dashed', linewidth=2)
        ax.text(media_amostral + 0.1, 0.35, f'Média = {media_amostral:.2f}', color='red', fontsize=12)
        
        st.pyplot(fig)

        desvio_padrao = np.std(medias)
        z_valor = (media_amostral - 0) / desvio_padrao
        
        st.write(f"Média das amostras: {media_amostral:.2f}")
        st.write(f"Desvio padrão das amostras: {desvio_padrao:.2f}")
        st.write(f"Valor de Z: {z_valor:.2f}")
        
        probabilidade = stats.norm.cdf(z_valor)
        st.write(f"Probabilidade associada ao Z: {probabilidade:.4f}")

    def coeficiente_covariancia(self):
        dados_x = np.random.normal(0, 1, self.tamanho_amostra)
        dados_y = 0.8 * dados_x + np.random.normal(0, 0.2, self.tamanho_amostra)  

        cov_xy = np.cov(dados_x, dados_y)[0, 1]
        rho = np.corrcoef(dados_x, dados_y)[0, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(dados_x, dados_y, color='mediumseagreen', alpha=0.7, edgecolor='black')
        ax.set_title("Dispersão entre X e Y", fontsize=16)
        ax.set_xlabel("Dados X", fontsize=12)
        ax.set_ylabel("Dados Y", fontsize=12)
        
        m, b = np.polyfit(dados_x, dados_y, 1)
        ax.plot(dados_x, m*dados_x + b, color='red', linewidth=2, linestyle='--')
        ax.text(0, -1.5, f'y = {m:.2f}x + {b:.2f}', color='red', fontsize=12)

        st.pyplot(fig)
        
        st.write(f"Covariância entre X e Y: {cov_xy:.2f}")
        st.write(f"Coeficiente de Correlação (rho) entre X e Y: {rho:.2f}")

    def teste_t_student(self, media_populacional=0):
        dados = np.random.normal(5, 1, self.tamanho_amostra)
        
        media_amostral = np.mean(dados)
        desvio_padrao = np.std(dados, ddof=1)
        t_stat = (media_amostral - media_populacional) / (desvio_padrao / np.sqrt(self.tamanho_amostra))
        
        p_valor = stats.t.sf(np.abs(t_stat), df=self.tamanho_amostra - 1) * 2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dados, bins=20, alpha=0.7, color='cornflowerblue', edgecolor='black')
        ax.set_title("Distribuição da Amostra (t-Student)", fontsize=16)
        ax.set_xlabel("Valores", fontsize=12)
        ax.set_ylabel("Frequência", fontsize=12)
        
        ax.axvline(media_amostral, color='red', linestyle='dashed', linewidth=2)
        ax.text(media_amostral + 0.2, 4, f'Média = {media_amostral:.2f}', color='red', fontsize=12)
        
        st.pyplot(fig)

        st.write(f"Estatística t: {t_stat:.2f}")
        st.write(f"p-valor: {p_valor:.4f}")
        
        nivel_significancia = 0.05
        if p_valor < nivel_significancia:
            st.write("Rejeitamos a hipótese nula: a média da amostra é significativamente diferente da média populacional.")
        else:
            st.write("Não rejeitamos a hipótese nula: não há evidências suficientes para afirmar que a média da amostra é diferente da média populacional.")

estatistica = Estatistica()

st.title("Dashboard Estatístico")
st.sidebar.header("Escolha o cálculo")

opcao = st.sidebar.selectbox(
    "Selecione uma análise:",
    ("Teorema Central do Limite", "Coeficiente de Covariância", "Teste t-Student")
)

tamanho_amostra = st.sidebar.slider("Tamanho da Amostra", 10, 100, 30)
n_amostras = st.sidebar.slider("Número de Amostras (apenas TCL)", 100, 5000, 1000)
media_populacional = st.sidebar.number_input("Média Populacional (para Teste t)", value=0.0)

estatistica.tamanho_amostra = tamanho_amostra
estatistica.n_amostras = n_amostras

if opcao == "Teorema Central do Limite":
    estatistica.teorema_central_limite()
elif opcao == "Coeficiente de Covariância":
    estatistica.coeficiente_covariancia()
elif opcao == "Teste t-Student":
    estatistica.teste_t_student(media_populacional=media_populacional)
