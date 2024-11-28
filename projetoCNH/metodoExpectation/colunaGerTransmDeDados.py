import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import streamlit as st

caminho_arquivo = r"C:\WS\scripts\projetoCNH\Lista_NPS_Positivo_V4(1).xlsx"

df = pd.read_excel(caminho_arquivo, usecols=["geração e transmissão de dados para gestão agrícola (csat)"])

dados_limpos = df.dropna()

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(dados_limpos)

clusters = gmm.predict(dados_limpos)

dados_limpos['Cluster'] = clusters

cluster_counts = dados_limpos['Cluster'].value_counts()

plt.figure(figsize=(10, 6))

cores = ['orange', 'green', 'blue']

for cluster in range(3):
    cluster_data = dados_limpos[dados_limpos['Cluster'] == cluster]
    plt.scatter(
        cluster_data.index,
        cluster_data["geração e transmissão de dados para gestão agrícola (csat)"],
        label=f"Cluster {cluster}",
        color=cores[cluster]
    )

plt.title("Distribuição dos Clusters na Geração e Transmissão de Dados para Gestão Agrícola (csat)")
plt.xlabel("Índice dos Dados")
plt.ylabel("Geração e Transmissão de Dados para Gestão Agrícola (csat)")
plt.legend()
plt.tight_layout()

grafico_path = "grafico_clusters_transmissao_dados_agricola.png"
plt.savefig(grafico_path)

st.title("Análise de Clusters com Expectation Maximization")
st.write("Este é um gráfico mostrando a distribuição dos clusters na coluna de geração e transmissão de dados para gestão agrícola (csat).")

st.image(grafico_path)

st.write("### Detalhes dos Clusters")
st.write(dados_limpos.groupby('Cluster').mean())

st.write("### Contagem de Pontos por Cluster")
for i, count in cluster_counts.items():
    st.write(f"Cluster {i}: {count} pontos")

st.write("### Caixa de Contagem de Clusters")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Cluster Laranja")
    st.metric("Pontos", cluster_counts.get(0, 0))

with col2:
    st.subheader("Cluster Verde")
    st.metric("Pontos", cluster_counts.get(1, 0))

with col3:
    st.subheader("Cluster Azul")
    st.metric("Pontos", cluster_counts.get(2, 0))

