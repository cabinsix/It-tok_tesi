# -*- coding: utf-8 -*-
"""
=====================================================================
ANALISI MONOVARIATA SUI RANGHI (Spearman + permutazione + Bonferroni)
=====================================================================
Input : risultati_grezzi.csv  (output dell'estrattore A LIVELLO DOCUMENTO)
Unita': UN VIDEO = UNA RIGA. NESSUN groupby: le feature sono gia' aggregate
        per documento dall'estrattore.
Corpora: trittok_gen + trittok_polso (i due sotto-corpus TrIt-Tok con metadati
         video, like_count/view_count/comment_count).
Target : comment_count.

Le emozioni positive/negative sono RICOMPOSTE dalle 8 colonne 'elita_avg_*'
prodotte dall'estrattore:
  positive = gioia + sorpresa + fiducia + aspettativa
  negative = tristezza + paura + disgusto + rabbia
=====================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_CSV   = "risultati_grezzi.csv"
INPUT_SEP   = ";"                      # <-- il nuovo file usa il punto e virgola
CORPORA     = ["trittok_gen", "trittok_polso"]
TARGET      = "comment_count"
OUTPUT_CSV  = "risultati_monovariati_ranghi_bonferroni.csv"
N_RESAMPLES = 10000
RANDOM_STATE = 42
MIN_N       = 5                        # minimo osservazioni per un test
MISSING_MAX = 0.5                      # scarta feature con >50% di NaN

# Colonne identificative / categoriali da NON testare come feature.
ID_COLS = {
    "corpus", "documento", "id_corpus",     # <-- 'documento' (non 'doc_id')
    "subcorpus", "simmetria", "direzionalita", "CST", "distanza",
    "contesto", "concezione", "genere",
    "n_tokens", "n_frasi", "n_tokens_punct",
    "comment_count",
}

# Le 8 componenti emotive gia' aggregate per documento (elita_avg_<nome>).
EMO_POS = ["elita_avg_gioia", "elita_avg_sorpresa",
           "elita_avg_fiducia", "elita_avg_aspettativa"]
EMO_NEG = ["elita_avg_tristezza", "elita_avg_paura",
           "elita_avg_disgusto", "elita_avg_rabbia"]


# ---------------------------------------------------------------------
# 1. CARICAMENTO
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, sep=INPUT_SEP)
print(f"Caricato {INPUT_CSV}: {len(df)} righe, {len(df.columns)} colonne.")

# Filtro sui due sotto-corpus richiesti
df = df[df["corpus"].isin(CORPORA)].copy()
print(f"Dopo filtro corpus {CORPORA}: {len(df)} video.")

if df.empty:
    raise SystemExit(
        "ERRORE: nessuna riga per i corpora richiesti. "
        "Controlla che l'estrattore NON abbia escluso trittok_gen/trittok_polso "
        "e che i nomi in 'corpus' coincidano."
    )

# ---------------------------------------------------------------------
# 2. CONVERSIONE NUMERICA ROBUSTA (virgola->punto dove serve)
# ---------------------------------------------------------------------
def to_numeric_col(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )

# Il target deve essere numerico
if TARGET not in df.columns:
    raise SystemExit(f"ERRORE: colonna target '{TARGET}' assente.")
df[TARGET] = to_numeric_col(df[TARGET])

# ---------------------------------------------------------------------
# 3. EMOZIONI POS/NEG A LIVELLO VIDEO (ricomposte dalle elita_avg_*)
# ---------------------------------------------------------------------
pos_present = [c for c in EMO_POS if c in df.columns]
for c in pos_present:
    df[c] = to_numeric_col(df[c])
df["emozioni_positive_video"] = df[pos_present].sum(axis=1) if pos_present else np.nan
print(f"[emo] emozioni positive ricomposte da {pos_present}")

neg_present = [c for c in EMO_NEG if c in df.columns]
for c in neg_present:
    df[c] = to_numeric_col(df[c])
df["emozioni_negative_video"] = df[neg_present].sum(axis=1) if neg_present else np.nan
print(f"[emo] emozioni negative ricomposte da {neg_present}")

# ---------------------------------------------------------------------
# 4. SELEZIONE DELLE FEATURE DA TESTARE
# ---------------------------------------------------------------------
# Convertiamo a numerico tutte le colonne candidate (escluse ID/target/emo grezze).
emo_raw = set(EMO_POS + EMO_NEG + [
    "elita_avg_positiva", "elita_avg_negativa",
    "elita_n_emotional", "elita_n_superlatives",
])
candidate = [c for c in df.columns
             if c not in ID_COLS and c not in emo_raw
             and c not in ("emozioni_positive_video", "emozioni_negative_video")]

for c in candidate:
    df[c] = to_numeric_col(df[c])

feature_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
# Aggiungiamo le due emozioni sintetiche di video
feature_cols += ["emozioni_positive_video", "emozioni_negative_video"]

print(f"Feature da testare: {len(feature_cols)}")

# ---------------------------------------------------------------------
# 5. SPEARMAN + TEST DI PERMUTAZIONE
# ---------------------------------------------------------------------
risultati = []
print(f"\nAnalisi sui ranghi (N={len(df)} video) con test di permutazione...")

def _spearman_stat(x, y):
    return stats.spearmanr(x, y).statistic

for col in feature_cols:
    if col not in df.columns:
        continue
    # varianza nulla o troppi missing -> salta
    if df[col].nunique(dropna=True) <= 1:
        continue
    if df[col].isna().sum() > len(df) * MISSING_MAX:
        continue

    data_model = df[[TARGET, col]].dropna()
    if len(data_model) < MIN_N:
        continue

    try:
        rho, p_asint = stats.spearmanr(data_model[col], data_model[TARGET])
        res_perm = stats.permutation_test(
            (data_model[col].values, data_model[TARGET].values),
            statistic=_spearman_stat,
            permutation_type="pairings",
            alternative="two-sided",
            n_resamples=N_RESAMPLES,
            random_state=RANDOM_STATE,
        )
        risultati.append({
            "Variabile": col,
            "N": len(data_model),
            "Spearman_Rho": rho,
            "p-value_Asintotico": p_asint,
            "p-value_Permutazione": res_perm.pvalue,
        })
    except Exception as e:
        print(f"  ! {col}: {e}")

df_res = pd.DataFrame(risultati)

# ---------------------------------------------------------------------
# 6. BONFERRONI + OUTPUT
# ---------------------------------------------------------------------
if df_res.empty:
    print("Nessun test eseguito (dati insufficienti).")
else:
    m = len(df_res)
    print(f"\nCorrezione di Bonferroni su {m} test.")
    df_res["p-value_Permutazione_Bonferroni"] = np.clip(
        df_res["p-value_Permutazione"] * m, 0.0, 1.0
    )
    df_res["Significativo_Bonferroni (p<0.05)"] = np.where(
        df_res["p-value_Permutazione_Bonferroni"] < 0.05, "Sì", "No"
    )
    df_res = df_res.sort_values(by="p-value_Permutazione_Bonferroni")

    print("\n--- RISULTATI (ranghi, Bonferroni) ---")
    print(df_res.to_string(index=False))
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvato in {OUTPUT_CSV}")