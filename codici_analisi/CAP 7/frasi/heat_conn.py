import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Caricamento del file CSV
# Sostituisci con il percorso corretto del tuo file se necessario (es. 'risultati_connettivi_gold.csv')
file_path = "risultati_connettivi_gold.csv"
df = pd.read_csv(file_path)

# Pulizia dei nomi dei corpora se contengono caratteri di escape (es. trittok\_gen -> trittok_gen)
df["Corpus"] = df["Corpus"].str.replace(r"\\", "", regex=True)

# 2. Creazione della matrice Pivot (Connettivo x Corpus)
# Usiamo la 'Percentuale' per confrontare equamente corpora di dimensioni diverse
pivot_df = df.pivot(index="Connettivo", columns="Corpus", values="Percentuale")

# Sostituiamo i valori NaN (connettivi assenti in un corpus) con 0
pivot_df = pivot_df.fillna(0)

# 3. Opzionale: Filtraggio per una migliore leggibilità
# Visualizza solo i connettivi che hanno una percentuale maggiore dell'1% in almeno un corpus
soglia_percentuale = 1.0
pivot_filtrato = pivot_df[pivot_df.max(axis=1) >= soglia_percentuale]

# Ordiniamo i connettivi in base alla media delle percentuali per metterli in ordine di importanza
pivot_filtrato = pivot_filtrato.reindex(
    pivot_filtrato.mean(axis=1).sort_values(ascending=False).index
)

# 4. Generazione del grafico
# Regola l'altezza (14) a seconda di quanti connettivi decidi di mostrare
plt.figure(figsize=(10, 14))

sns.heatmap(
    pivot_filtrato,
    annot=True,  # Mostra i valori percentuali nelle celle
    fmt=".2f",  # Formato a due decimali
    cmap="YlOrRd",  # Gradazione di colore dal giallo (basso) al rosso (alto)
    linewidths=0.5,  # Sottile linea di separazione tra le celle
    cbar_kws={"label": "Percentuale (%)"},
)

plt.title(
    f"Confronto Connettivi tra Corpora (Soglia >= {soglia_percentuale}%)",
    fontsize=16,
    pad=20,
    weight="bold",
)
plt.ylabel("Connettivo", fontsize=12, weight="bold")
plt.xlabel("Corpus", fontsize=12, weight="bold")
plt.xticks(
    rotation=45, ha="right"
)  # Ruota i nomi dei corpora per non farli sovrapporre
plt.tight_layout()

# 5. Mostra la heatmap
plt.show()