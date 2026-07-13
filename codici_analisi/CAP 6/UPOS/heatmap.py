import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. Dati Assoluti estratti dalle tabelle (frequenze grezze)
data_assoluti = {
    "UPOS": [
        "NOUN", "VERB", "ADP", "ADJ", "DET", "PROPN", "ADV", 
        "PRON", "AUX", "CCONJ", "NUM", "SCONJ", "X", "INTJ", "SYM", "PART"
    ],
    "ISDT": [14544, 7066, 11490, 4978, 11435, 4652, 4132, 3482, 3131, 2077, 1387, 891, 59, 56, 7, 3],
    "ParlaMint": [4216, 2040, 3065, 1131, 3219, 176, 1076, 1139, 986, 583, 243, 390, 4, 4, 0, 0],
    "PolSo": [6401, 4466, 4479, 2041, 3856, 749, 2989, 3081, 2152, 1275, 233, 1146, 11, 137, 14, 0],
    "Ittok": [12221, 9861, 8439, 3780, 7793, 1186, 7234, 7081, 4554, 2787, 539, 2374, 40, 348, 43, 0],
    "Gen": [5820, 5395, 3960, 1739, 3937, 437, 4245, 4000, 2402, 1512, 306, 1228, 29, 211, 29, 0],
    "KIParla": [2668, 2384, 1903, 947, 2194, 424, 2212, 1882, 1044, 1002, 172, 698, 354, 791, 4, 1],
}

df_assoluti = pd.DataFrame(data_assoluti).set_index("UPOS")

# 2. Calcolo della Somiglianza Coseno direttamente sui numeri assoluti
matrice_somiglianza = cosine_similarity(df_assoluti.T)

# Creazione DataFrame per visualizzazione in console
corpora_names = df_assoluti.columns
df_simil = pd.DataFrame(
    matrice_somiglianza, index=corpora_names, columns=corpora_names
)

print("--- MATRICE DI SOMIGLIANZA COSENO (CALCOLATA SUI NUMERI ASSOLUTI) ---")
print(df_simil.round(4))

# 3. Generazione della Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df_simil, annot=True, cmap="YlGnBu", vmin=0.85, vmax=1.0, fmt=".3f")
plt.title("Somiglianza Sintattica Globale tra i Corpora\n(Basata sulle frequenze assolute UPOS)")
plt.tight_layout()
plt.show()