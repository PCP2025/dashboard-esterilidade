import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import sem

# Título e introdução
st.title("Análise de Risco - Bolsas de Plasma")
st.markdown("""
Este dashboard mostra a probabilidade de **aprovação das bolsas** de plasma em função do tempo de armazenamento até a análise de esterilidade, com base em:
- **Regressão Logística** (linha azul)
- **Média móvel** (linha laranja)
""")

# Upload do arquivo CSV
df = pd.read_csv("Binarios.csv", sep=';')
df.columns = df.columns.str.strip()
df = df[df['Resultado'].isin(['0', '1'])]
df['Resultado'] = df['Resultado'].astype(int)
df['Dias'] = pd.to_numeric(df['Dias'], errors='coerce')
df = df.dropna()

# Modelagem
X = df[['Dias']]
y = df['Resultado']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# Previsões
dias_range = np.linspace(df['Dias'].min(), df['Dias'].max(), 300).reshape(-1, 1)
dias_range_scaled = scaler.transform(dias_range)
proba_pred = model.predict_proba(dias_range_scaled)[:, 1]

# Sliders interativos
meta = st.slider("Meta mínima de aprovação (%)", 50, 99, 90) / 100
bin_size = st.slider("Tamanho da janela para média móvel (dias):", 10, 60, 20)

# Cálculo do ponto de corte baseado na meta
dias_meta = dias_range[proba_pred >= meta]
ponto_meta = dias_meta[-1] if len(dias_meta) > 0 else None

# Média móvel
df['bin'] = (df['Dias'] // bin_size) * bin_size
media_movel = df.groupby('bin')['Resultado'].mean()

# Faixas de risco (baseadas somente na regressão logística)
def faixa_risco_dinamica(p, meta):
    if p >= meta:
        return 'Baixo risco'
    elif p >= 0.5:
        return 'Médio risco'
    else:
        return 'Alto risco'

riscos = [faixa_risco_dinamica(p, meta) for p in proba_pred]

# Plotagem
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dias_range, proba_pred, label='Regressão logística (tendência)', color='blue', linewidth=2)
ax.plot(media_movel.index, media_movel, label='Média móvel', color='orange', linestyle='--')

# Cálculo do erro padrão e limites de confiança
#erro_padrao = df.groupby('bin')['Resultado'].apply(sem)
#conf_sup = media_movel + 1.96 * erro_padrao
#conf_inf = media_movel - 1.96 * erro_padrao

# Preenchendo a área do intervalo de confiança
#ax.fill_between(media_movel.index, conf_inf, conf_sup, color='orange', alpha=0.2, label='Intervalo de confiança (95%)')

# Faixas visuais com base na regressão logística
for i in range(1, len(dias_range)):
    cor = {
        'Baixo risco': '#A8E6A1',
        'Médio risco': '#FFF3B0',
        'Alto risco': '#FFB3B3'
    }[riscos[i]]
    ax.axvspan(dias_range[i-1][0], dias_range[i][0], facecolor=cor, alpha=0.2)

# Linha de corte
if ponto_meta:
    ax.axvline(ponto_meta, color='red', linestyle='--', label=f'Corte para {meta*100:.0f}%: {int(ponto_meta)} dias')

# Finalização do gráfico
ax.set_title('Resultados Esterilidade vs Tempo de Vida até a Análise')
ax.set_xlabel('Dias de armazenamento')
ax.set_ylabel('Probabilidade estimada')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Feedback textual
if ponto_meta:
    st.markdown(f"🔴 O **ponto de corte** para manter aprovação acima de {meta*100:.0f}% é cerca de **{int(ponto_meta)} dias**.")
else:
    st.markdown("❌ Nenhum ponto com essa taxa mínima de aprovação foi encontrado.")

# Legenda de risco
st.markdown(f"""
🔹 **Critérios de risco (baseados na meta de {int(meta * 100)}%):**  
- 🟢 Baixo risco: aprovação ≥ {int(meta * 100)}%  
- 🟡 Médio risco: entre 50% e {int(meta * 100)}%  
- 🔴 Alto risco: < 50%
""")

# Após treinar o modelo:
model.fit(X_scaled, y)
# Coeficientes (para uso interno)
coef = model.coef_[0][0]
intercept = model.intercept_[0]
print(f"Coeficiente: {coef:.4f}")
print(f"Intercepto: {intercept:.4f}")


