import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px

caminho_arquivo = r"C:\WS\scripts\projetoCNH\Lista_NPS_Positivo_V4(1).xlsx"

coluna = "geração e transmissão de dados para gestão agrícola (csat)"

st.title("Análise de Clusters com K-Means")
st.write(f"Este app analisa a coluna **{coluna}** do arquivo Excel aplicando o método K-Means para agrupamento.")

df = pd.read_excel(caminho_arquivo, usecols=[coluna])

dados_limpos = df.dropna()

num_clusters = st.slider("Número de Clusters (K)", min_value=2, max_value=6, value=3, step=1)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(dados_limpos)

dados_limpos['Cluster'] = clusters

st.write("### Resultados do Agrupamento")
st.write(dados_limpos)

cluster_counts = dados_limpos['Cluster'].value_counts()

st.write("### Quantidade de Pontos por Cluster")
st.write(cluster_counts.reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Quantidade'}))

fig = px.bar(
    x=cluster_counts.index, 
    y=cluster_counts.values, 
    labels={'x': 'Cluster', 'y': 'Quantidade'}, 
    title="Contagem de Pontos por Cluster"
)
fig.update_traces(text=cluster_counts.values, textposition='outside')  # Adicionar rótulos
st.plotly_chart(fig)

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

grafico_path = "grafico_clusters_kmeans.png"
plt.savefig(grafico_path)

st.image(grafico_path)

st.write("### Coordenadas dos Centróides")
st.write(pd.DataFrame(kmeans.cluster_centers_, columns=[coluna]))
