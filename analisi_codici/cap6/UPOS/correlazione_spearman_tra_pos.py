import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Dati percentuali aggiornati
data_percentuali = {
    "UPOS": ["NOUN", "VERB", "ADP", "ADJ", "DET", "PROPN", "ADV", "PRON", "AUX", "CCONJ", "NUM", "SCONJ", "X", "INTJ", "SYM", "PART"],
    "ISDT": [20.96, 10.18, 16.56, 7.17, 16.48, 6.70, 5.95, 5.02, 4.51, 2.99, 2.00, 1.28, 0.09, 0.08, 0.01, 0.00],
    "ParlaMint": [23.07, 11.16, 16.77, 6.19, 17.62, 0.96, 5.89, 6.23, 5.40, 3.19, 1.33, 2.13, 0.02, 0.02, 0.00, 0.00],
    "PolSo": [19.38, 13.52, 13.56, 6.18, 11.67, 2.27, 9.05, 9.33, 6.52, 3.86, 0.71, 3.47, 0.03, 0.41, 0.04, 0.00],
    "Ittok": [17.90, 14.44, 12.36, 5.54, 11.41, 1.74, 10.59, 10.37, 6.67, 4.08, 0.79, 3.48, 0.06, 0.51, 0.06, 0.00],
    "Gen": [16.51, 15.30, 11.23, 4.93, 11.17, 1.24, 12.04, 11.35, 6.81, 4.29, 0.87, 3.48, 0.08, 0.60, 0.08, 0.00],
    "KIParla": [14.28, 12.76, 10.19, 5.07, 11.75, 2.27, 11.84, 10.07, 5.59, 5.36, 0.92, 3.74, 1.90, 4.23, 0.02, 0.01],
}

df_pct = pd.DataFrame(data_percentuali).set_index("UPOS")

# 2. Selezioniamo SOLO le 11 UPOS presenti nella tua immagine, nell'ordine esatto
upos_selezionate = ["NOUN", "VERB", "ADP", "DET", "ADV", "PRON", "AUX", "ADJ", "CCONJ", "SCONJ", "PROPN"]
df_filtered = df_pct.loc[upos_selezionate]

# Trasponiamo per avere i corpora sulle righe e calcoliamo spearman
matrice_spearman = df_filtered.T.corr(method="spearman")

# 3. Creiamo la "maschera" per nascondere il triangolo inferiore
# k=-1 nasconde tutto ciò che sta rigorosamente sotto la diagonale principale
mask = np.tril(np.ones_like(matrice_spearman, dtype=bool), k=-1)

# 4. Generazione del grafico
plt.figure(figsize=(10, 8))

# Disegniamo la heatmap applicando maschera e colori specifici
sns.heatmap(
    matrice_spearman, 
    mask=mask,               # Nasconde il triangolo inferiore
    annot=True,              # Mostra i numeri
    cmap="PiYG_r",           # Mappa colori Rosa (+1) e Verde (-1)
    vmin=-1,                 # Valore minimo scala
    vmax=1,                  # Valore massimo scala
    fmt=".2g",               # Formatta i numeri per rimuovere gli zeri finali superflui (es. -0.61 invece di -0.610)
    linewidths=0             # Nessun bordo tra le celle, come nella tua immagine
)

# Mettiamo le etichette dell'asse X in alto per replicare l'immagine originale
plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
plt.xticks(rotation=90) # Ruotiamo le etichette in alto di 90 gradi

# Rimuoviamo i nomi degli assi "UPOS" per pulizia
plt.xlabel('')
plt.ylabel('')

plt.tight_layout()
plt.show()