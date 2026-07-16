import pandas as pd
import re

file_metadati = 'ITTOK_merged_allineato_con_metadati.csv'
file_disfluenze = 'disfluenze_per_frase.csv'

print("1. Caricamento dei file originali...")

df_metadati = pd.read_csv(file_metadati, sep=';', low_memory=False)
df_disfluenze = pd.read_csv(file_disfluenze, sep=',', low_memory=False)

# Colonne delle disfluenze
colonne_disf_orig = [
    'ripetizioni',
    'reparandum',
    'pause',
    'prolungamenti',
    'pause_piene',
    'segnali_discorsivi'
]

# Rinominare per evitare conflitti
mappa_rinomina = {col: f"disf_{col}" for col in colonne_disf_orig}
df_disfluenze = df_disfluenze.rename(columns=mappa_rinomina)

col_disfluenze = [f"disf_{col}" for col in colonne_disf_orig]

# Conversione in numerico
for col in col_disfluenze:
    df_disfluenze[col] = (
        pd.to_numeric(df_disfluenze[col], errors='coerce')
        .fillna(0)
    )

###########################################################
# Conteggio parole
###########################################################

print("2. Conteggio parole...")

df_disfluenze["n_parole"] = (
    df_disfluenze["text"]
    .fillna("")
    .astype(str)
    .apply(lambda x: len(x.split()))
)

###########################################################
# Estrazione doc_key
###########################################################

def estrai_doc_key(valore):
    val = str(valore).strip()
    match = re.search(r'([a-zA-Z]\d+(?:_[a-zA-Z0-9]+)?)', val)
    if match:
        return match.group(1).lower()
    return "unknown"

print("3. Generazione doc_key...")

if 'doc_id' in df_metadati.columns:
    df_metadati['doc_key'] = df_metadati['doc_id'].apply(estrai_doc_key)
else:
    col_rif = 'file' if 'file' in df_metadati.columns else 'sent_id'
    df_metadati['doc_key'] = df_metadati[col_rif].apply(estrai_doc_key)

df_disfluenze['doc_key'] = df_disfluenze['file'].apply(estrai_doc_key)

###########################################################
# Aggregazione
###########################################################

print("4. Aggregazione delle disfluenze per documento...")

# Media
df_media = (
    df_disfluenze
    .groupby("doc_key")[col_disfluenze]
    .mean()
    .reset_index()
)

# Somma
df_somma = (
    df_disfluenze
    .groupby("doc_key")[col_disfluenze]
    .sum()
    .reset_index()
)

# Numero totale parole
df_parole = (
    df_disfluenze
    .groupby("doc_key")["n_parole"]
    .sum()
    .reset_index()
)

###########################################################
# Frequenze normalizzate
###########################################################

print("5. Calcolo frequenze normalizzate (per 100 parole)...")

df_norm = pd.merge(df_somma, df_parole, on="doc_key")

for col in col_disfluenze:
    df_norm[col] = (
        df_norm[col] / df_norm["n_parole"] * 100
    )

df_norm = df_norm.drop(columns=["n_parole"])

###########################################################
# Merge
###########################################################

print("6. Merge con i metadati...")

df_media_all = pd.merge(
    df_metadati,
    df_media,
    on="doc_key",
    how="inner"
)

df_somma_all = pd.merge(
    df_metadati,
    df_somma,
    on="doc_key",
    how="inner"
)

df_norm_all = pd.merge(
    df_metadati,
    df_norm,
    on="doc_key",
    how="inner"
)

print(f"Documenti allineati: {len(df_media_all)}")

if len(df_media_all) == 0:
    print("ERRORE: merge vuoto.")
    exit()

###########################################################
# Pianificazione
###########################################################

for df in [df_media_all, df_somma_all, df_norm_all]:
    df["pianificazione"] = pd.to_numeric(
        df["pianificazione"],
        errors="coerce"
    )

###########################################################
# Tabelle finali
###########################################################

tab_media = (
    df_media_all
    .groupby("pianificazione")[col_disfluenze]
    .mean()
    .sort_index()
)

tab_somma = (
    df_somma_all
    .groupby("pianificazione")[col_disfluenze]
    .mean()
    .sort_index()
)

tab_norm = (
    df_norm_all
    .groupby("pianificazione")[col_disfluenze]
    .mean()
    .sort_index()
)

###########################################################
# Pulizia nomi colonne
###########################################################

rinomina = {
    c: c.replace("disf_", "")
    for c in col_disfluenze
}

tab_media = tab_media.rename(columns=rinomina)
tab_somma = tab_somma.rename(columns=rinomina)
tab_norm = tab_norm.rename(columns=rinomina)

###########################################################
# Output
###########################################################

print("\n" + "="*80)
print("MEDIA DELLE DISFLUENZE PER DOCUMENTO")
print("="*80)
print(tab_media)

print("\n" + "="*80)
print("SOMMA DELLE DISFLUENZE PER DOCUMENTO")
print("="*80)
print(tab_somma)

print("\n" + "="*80)
print("DISFLUENZE NORMALIZZATE (OGNI 100 PAROLE)")
print("="*80)
print(tab_norm)

###########################################################
# Salvataggio CSV
###########################################################

tab_media.to_csv(
    "media_disfluenze_per_pianificazione.csv",
    sep=";",
    encoding="utf-8-sig"
)

tab_somma.to_csv(
    "somma_disfluenze_per_pianificazione.csv",
    sep=";",
    encoding="utf-8-sig"
)

tab_norm.to_csv(
    "disfluenze_normalizzate_per_pianificazione.csv",
    sep=";",
    encoding="utf-8-sig"
)

print("\nFile salvati:")
print("- media_disfluenze_per_pianificazione.csv")
print("- somma_disfluenze_per_pianificazione.csv")
print("- disfluenze_normalizzate_per_pianificazione.csv")

from scipy.stats import spearmanr

print("\nCORRELAZIONE DI SPEARMAN")

for col in col_disfluenze:
    rho, p = spearmanr(
        df_media_all["pianificazione"],
        df_media_all[col],
        nan_policy="omit"
    )

    print(f"{col:25s} rho={rho:.3f}   p={p:.4f}")

    import statsmodels.api as sm

print("\nREGRESSIONE LINEARE\n")


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,7))

etichette = {
    "disf_ripetizioni":"Ripetizioni",
    "disf_reparandum":"Reparandum",
    "disf_pause":"Pause",
    "disf_prolungamenti":"Prolungamenti",
    "disf_pause_piene":"Pause piene",
    "disf_segnali_discorsivi":"Segnali discorsivi"
}

for col in col_disfluenze:

    dati = df_media_all[["pianificazione", col]].dropna()

    x = dati["pianificazione"]
    y = dati[col]

    # retta di regressione
    coeff = np.polyfit(x, y, 1)
    p = np.poly1d(coeff)

    xx = np.linspace(x.min(), x.max(), 100)

    plt.plot(xx, p(xx), linewidth=2, label=etichette[col])

plt.xlabel("Livello di pianificazione")
plt.ylabel("Media delle disfluenze")
plt.title("Andamento delle disfluenze al variare della pianificazione")

plt.xticks(sorted(df_media_all["pianificazione"].dropna().unique()))
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

from scipy.stats import kruskal
import scikit_posthocs as sp

print("\n" + "="*80)
print("TEST DI KRUSKAL-WALLIS")
print("="*80)

for col in col_disfluenze:

    gruppi = []

    for livello in sorted(df_media_all["pianificazione"].dropna().unique()):
        valori = (
            df_media_all[df_media_all["pianificazione"] == livello][col]
            .dropna()
        )
        gruppi.append(valori)

    H, p = kruskal(*gruppi)

    print(f"\n{col}")
    print(f"H = {H:.3f}   p = {p:.4f}")

    # Se significativo, esegui Dunn
    if p < 0.05:

        print("\nPost-hoc di Dunn (correzione Holm):")

        posthoc = sp.posthoc_dunn(
            df_media_all,
            val_col=col,
            group_col="pianificazione",
            p_adjust="holm"
        )

        print(posthoc.round(4))