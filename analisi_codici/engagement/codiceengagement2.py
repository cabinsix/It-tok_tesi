import numpy as np
import pandas as pd
from scipy import stats  # <--- Fondamentale per i test non-parametrici e permutazioni

# 1. Caricamento del dataset
df = pd.read_csv("merged_allineato_con_metadati.csv", sep=";")

# 2. Filtraggio per il corpus specifico
df_filtered = df[df["corpus"] == "tr-it-tok"].copy()

# --- Pulizia e Conversione delle colonne critiche in numeri ---
colonne_da_convertire = [
    "mdd",
    "mean_P_w",
    "mean_D_w",
    "mean_V_w",
    "mean_sat_rate",
]

for col in colonne_da_convertire:
    if col in df_filtered.columns:
        df_filtered[col] = (
            df_filtered[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

# 3. Definizione delle logiche di aggregazione
emozioni_positive = ["avg_fiducia", "avg_gioia", "avg_aspettativa", "avg_sorpresa"]
emozioni_negative = ["avg_rabbia", "avg_paura", "avg_disgusto", "avg_tristezza"]

for col in emozioni_positive + emozioni_negative:
    if col in df_filtered.columns:
        df_filtered[col] = (
            df_filtered[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

df_filtered["media_pos_frase"] = df_filtered[emozioni_positive].mean(axis=1)
df_filtered["media_neg_frase"] = df_filtered[emozioni_negative].mean(axis=1)

variabili_media = [
    "media_pos_frase",
    "media_neg_frase",
    "mdd",
    "mean_P_w",
    "mean_D_w",
    "mean_V_w",
    "mean_sat_rate",
]

escluse = (
    [
        "unit_id",
        "sent_id",
        "corpus",
        "doc_id",
        "id_corpus",
        "comment_count",
    ]
    + emozioni_positive
    + emozioni_negative
    + variabili_media
)

variabili_conteggio = [
    col
    for col in df_filtered.columns
    if col not in escluse and pd.api.types.is_numeric_dtype(df_filtered[col])
]

agg_dict = {col: "sum" for col in variabili_conteggio}
for col in variabili_media:
    agg_dict[col] = "mean"

agg_dict["comment_count"] = "first"

# 4. Aggregazione a livello di Video (doc_id)
df_video = df_filtered.groupby("doc_id").agg(agg_dict).reset_index()

df_video = df_video.rename(
    columns={
        "media_pos_frase": "emozioni_positive_video",
        "media_neg_frase": "emozioni_negative_video",
    }
)

feature_cols = (
    variabili_conteggio
    + ["emozioni_positive_video", "emozioni_negative_video"]
    + [c for c in variabili_media if c not in ["media_pos_frase", "media_neg_frase"]]
)

# -------------------------------------------------------------------------
# NUOVO 5. Analisi Non-Parametrica: Spearman + Test di Permutazione Esatto
# -------------------------------------------------------------------------
risultati_ranghi = []
target = "comment_count"

print(
    f"Inizio analisi sui ranghi (N={len(df_video)} video) con Test di Permutazione..."
)

for col in feature_cols:
    # Salta la colonna se ha varianza zero o troppi missing values
    if df_video[col].nunique() <= 1 or df_video[col].isna().sum() > (
        len(df_video) * 0.5
    ):
        continue

    try:
        # Pulizia localizzata dei dati mancanti per la coppia di variabili
        data_model = df_video[[target, col]].dropna()

        if len(data_model) < 5:  # Troppi pochi dati rimasti
            continue

        # A. Calcolo del coefficiente di Spearman e del suo p-value asintotico
        rho, p_spearman = stats.spearmanr(data_model[col], data_model[target])

        # B. Calcolo del p-value esatto tramite Test di Permutazione (10.000 simulazioni)
        res_perm = stats.permutation_test(
            (data_model[col], data_model[target]),
            statistic=lambda x, y: stats.spearmanr(x, y)[0],
            permutation_type="pairings",
            alternative="two-sided",
            n_resamples=10000,
            random_state=42,
        )
        p_permutation = res_perm.pvalue

        risultati_ranghi.append(
            {
                "Variabile": col,
                "Spearman_Rho": rho,
                "p-value_Asintotico": p_spearman,
                "p-value_Permutazione": p_permutation,
                "Significativo (p<0.05)": "Sì" if p_permutation < 0.05 else "No",
            }
        )
    except Exception as e:
        print(f"Impossibile calcolare i ranghi per {col}: {e}")

# -------------------------------------------------------------------------
# NUOVO 6. Ordinamento, Correzione di Bonferroni e Output dei Risultati
# -------------------------------------------------------------------------
# 1. Creiamo il DataFrame con TUTTI i dati originali che avevi già inserito
df_risultati = pd.DataFrame(risultati_ranghi)

if not df_risultati.empty:
    # 2. Calcoliamo il numero totale di test effettuati (m)
    m_tests = len(df_risultati)
    print(f"\nApplicazione della correzione di Bonferroni su {m_tests} test effettuati.")

    # 3. Moltiplichiamo il p-value di permutazione per il numero di test (m), limitando il massimo a 1.0
    df_risultati["p-value_Permutazione_Bonferroni"] = np.clip(
        df_risultati["p-value_Permutazione"] * m_tests, 0.0, 1.0
    )

    # 4. Aggiungiamo la nuova colonna di significatività basata sul p-value corretto
    df_risultati["Significativo_Bonferroni (p<0.05)"] = np.where(
        df_risultati["p-value_Permutazione_Bonferroni"] < 0.05, "Sì", "No"
    )

    # 5. Ordiniamo i risultati (puoi scegliere se ordinare per il vecchio p-value o per il nuovo, l'ordine resta identico)
    df_risultati = df_risultati.sort_values(by="p-value_Permutazione_Bonferroni")

print("\n--- RISULTATI DEI TEST SUI RANGHI CON CORREZIONE DI BONFERRONI ---")
print(df_risultati.to_string(index=False))

# Salva i risultati in CSV (mantenendo la struttura precedente + le 2 nuove colonne)
df_risultati.to_csv("risultati_monovariati_ranghi_bonferroni.csv", index=False)
print("\nRisultati salvati con successo in 'risultati_monovariati_ranghi_bonferroni.csv'")