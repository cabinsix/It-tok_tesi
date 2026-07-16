import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carica il file contenente i dati di tutte le singole finestre
file_finestre = "mattr_lemmi_finestre500.csv"

try:
    df_windows = pd.read_csv(file_finestre)
except FileNotFoundError:
    print(
        f"❌ Errore: Assicurati di aver eseguito prima il tuo script per generare il file '{file_finestre}'"
    )
    exit()

# 2. Calcola l'ordine delle sottocartelle in base alla MEDIANA del MATTR
# 'ascending=False' ordina dal corpus con più ricchezza lessicale a quello con meno ricchezza
ordine_cartelle = (
    df_windows.groupby("cartella")["mattr"]
    .median()
    .sort_values(ascending=False)
    .index
)

# 3. Configurazione estetica del grafico
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")  # Sfondo a griglia bianca pulito

# 4. Generazione del Boxplot
sns.boxplot(
    data=df_windows,
    x="cartella",
    y="mattr",
    order=ordine_cartelle,  # Applica l'ordinamento calcolato prima
    palette="YlGnBu_r",  # Sfumatura di colori coerente (puoi cambiarla in 'Set2' o 'viridis')
    width=0.5,  # Spessore dei box per renderli più eleganti
    fliersize=3,  # Dimensione dei punti outlier
)

# 5. Ottimizzazione delle etichette e dei testi
plt.title(
    "Distribuzione del MATTR per Sottocartella\n(Finestra mobile = 500 token, ordinato per mediana decrescente)",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Sottocartella / Corpus", fontsize=12, labelpad=10)
plt.ylabel("Moving-Average Type-Token Ratio (MATTR)", fontsize=12, labelpad=10)

# Ruota i nomi delle sottocartelle per evitare che si sovrappongano
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Mostra il grafico
plt.tight_layout()
plt.show()