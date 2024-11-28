import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import streamlit as st

# Caminho do arquivo Excel
caminho_arquivo = r"C:\WS\scripts\projetoCNH\Lista_NPS_Positivo_V4(1).xlsx"

# Carregar o Excel, selecionando a nova coluna
df = pd.read_excel(caminho_arquivo, usecols=["adequação as diversas operações e implementos (csat)"])

# Remover valores ausentes (NaN)
dados_limpos = df.dropna()

# Aplicar o modelo Expectation Maximization
gmm = GaussianMixture(n_components=3, random_state=42)  # 3 clusters (ajustável)
gmm.fit(dados_limpos)

# Prever os clusters
clusters = gmm.predict(dados_limpos)

# Adicionar os clusters ao DataFrame para visualização
dados_limpos['Cluster'] = clusters

# Contagem de pontos em cada cluster
cluster_counts = dados_limpos['Cluster'].value_counts()

# Criar gráficos de distribuição dos clusters
plt.figure(figsize=(10, 6))

# Definir cores para os clusters
cores = ['orange', 'green', 'blue']

# Plotar os clusters
for cluster in range(3):  # 3 clusters (ajustável)
    cluster_data = dados_limpos[dados_limpos['Cluster'] == cluster]
    plt.scatter(
        cluster_data.index,   # Eixo X: índice dos dados
        cluster_data["adequação as diversas operações e implementos (csat)"],  # Eixo Y: valores da coluna selecionada
        label=f"Cluster {cluster}",
        color=cores[cluster]  # Usando as cores definidas
    )

plt.title("Distribuição dos Clusters na Adequação das Operações e Implementos")
plt.xlabel("Índice dos Dados")
plt.ylabel("Adequação das Operações e Implementos (csat)")
plt.legend()
plt.tight_layout()

# Salvar o gráfico temporariamente
plt.savefig("grafico_clusters_adequacao.png")

# Streamlit para visualização
st.title("Análise de Clusters com Expectation Maximization")
st.write("Este é um gráfico mostrando a distribuição dos clusters na coluna de adequação das operações e implementos.")

# Exibir o gráfico no Streamlit
st.image("grafico_clusters_adequacao.png")

# Detalhamento dos clusters
st.write("### Detalhes dos Clusters")
st.write(dados_limpos.groupby('Cluster').mean())

# Exibir a contagem de pontos em cada cluster
st.write("### Contagem de Pontos por Cluster")
for i, count in cluster_counts.items():
    st.write(f"Cluster {i}: {count} pontos")

# Criar caixas para mostrar a contagem de pontos em cada cluster
st.write("### Caixa de Contagem de Clusters")
col1, col2, col3 = st.columns(3)  # Dividir a tela em 3 colunas para os clusters

# Exibir os contadores nas caixas
with col1:
    st.subheader("Cluster Laranja")
    st.metric("Pontos", cluster_counts.get(0, 0))  # Contagem do Cluster 0 (laranja)

with col2:
    st.subheader("Cluster Verde")
    st.metric("Pontos", cluster_counts.get(1, 0))  # Contagem do Cluster 1 (verde)

with col3:
    st.subheader("Cluster Azul")
    st.metric("Pontos", cluster_counts.get(2, 0))  # Contagem do Cluster 2 (azul)
