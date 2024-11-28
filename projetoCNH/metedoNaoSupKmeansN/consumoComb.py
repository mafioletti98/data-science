import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px

# Caminho do arquivo Excel
caminho_arquivo = r"C:\WS\scripts\projetoCNH\Lista_NPS_Positivo_V4(1).xlsx"

# Nome da coluna para análise
coluna = "consumo de combustível (litros por hectares) (csat)"

# Título do app
st.title("Análise de Clusters com K-Means")
st.write(f"Este app analisa a coluna **{coluna}** do arquivo Excel aplicando o método K-Means para agrupamento.")

# Carregar a coluna
df = pd.read_excel(caminho_arquivo, usecols=[coluna])

# Remover valores ausentes
dados_limpos = df.dropna()

# Configuração do número de clusters (k)
num_clusters = st.slider("Número de Clusters (K)", min_value=2, max_value=6, value=3, step=1)

# Aplicar o K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(dados_limpos)

# Adicionar os clusters ao DataFrame
dados_limpos['Cluster'] = clusters

# Exibir os clusters
st.write("### Resultados do Agrupamento")
st.write(dados_limpos)

# Contagem de pontos por cluster
cluster_counts = dados_limpos['Cluster'].value_counts()

# Exibir a contagem diretamente no Streamlit
st.write("### Quantidade de Pontos por Cluster")
st.write(cluster_counts.reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Quantidade'}))

# Gráfico de barras com rótulos no topo
fig = px.bar(
    x=cluster_counts.index, 
    y=cluster_counts.values, 
    labels={'x': 'Cluster', 'y': 'Quantidade'}, 
    title="Contagem de Pontos por Cluster"
)
fig.update_traces(text=cluster_counts.values, textposition='outside')  # Adicionar rótulos
st.plotly_chart(fig)

# Gráfico de clusters
plt.figure(figsize=(10, 6))
cores = ['orange', 'green', 'blue', 'red', 'purple', 'brown']  # Até 6 clusters
for cluster in range(num_clusters):
    cluster_data = dados_limpos[dados_limpos['Cluster'] == cluster]
    plt.scatter(
        cluster_data.index,  # Índice dos dados no eixo X
        cluster_data[coluna],  # Valores da coluna no eixo Y
        label=f"Cluster {cluster}",
        color=cores[cluster]
    )

plt.title(f"Clusters na Coluna: {coluna}")
plt.xlabel("Índice dos Dados")
plt.ylabel(coluna)
plt.legend()
plt.tight_layout()

# Salvar o gráfico temporariamente
grafico_path = "grafico_clusters_kmeans.png"
plt.savefig(grafico_path)

# Exibir o gráfico no Streamlit
st.image(grafico_path)

# Exibir os centróides
st.write("### Coordenadas dos Centróides")
st.write(pd.DataFrame(kmeans.cluster_centers_, columns=[coluna]))
