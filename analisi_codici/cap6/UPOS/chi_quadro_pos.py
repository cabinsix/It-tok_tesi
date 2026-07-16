import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# 1. Dati perfettamente allineati riga per riga
raw_data = {
    "UPOS": [
        "NOUN",
        "ADP",
        "VERB",
        "DET",
        "PRON",
        "ADV",
        "AUX",
        "ADJ",
        "CCONJ",
        "SCONJ",
        "PROPN",
        "NUM",
        "INTJ",
        "SYM",
    ],
    "Polso_assolute": [
        6401,
        4479,
        4466,
        3856,
        3081,
        2989,
        2152,
        2041,
        1275,
        1146,
        749,
        233,
        137,
        14,
    ],
    "Gen_assolute": [
        5820,
        3960,
        5395,
        3937,
        4000,
        4245,
        2402,
        1739,
        1512,
        1228,
        437,
        306,
        211,
        29,
    ],
}

df = pd.DataFrame(raw_data)

# 2. Calcolo dei totali reali basati sulla somma delle colonne
TOTAL_POLSO = df["Polso_assolute"].sum()
TOTAL_GEN = df["Gen_assolute"].sum()

print(f"--- VERIFICA TOTALI ---")
print(f"Totale token PolSo: {TOTAL_POLSO}")
print(f"Totale token Gen: {TOTAL_GEN}\n")

p_values = []

# 3. Test Chi-Quadrato
for idx, row in df.iterrows():
    upos_polso = row["Polso_assolute"]
    upos_gen = row["Gen_assolute"]

    altre_polso = TOTAL_POLSO - upos_polso
    altre_gen = TOTAL_GEN - upos_gen

    contingency_table = [[upos_polso, upos_gen], [altre_polso, altre_gen]]

    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    p_values.append(p_val)

df["p-value_grezzo"] = p_values

# 4. Correzione di Benjamini-Hochberg per confronti multipli
reject, p_adjusted, _, _ = multipletests(
    df["p-value_grezzo"], alpha=0.05, method="fdr_bh"
)

df["p-value_corretto"] = p_adjusted
df["Significativo (FDR < 0.05)"] = [
    "Sì" if r else "No" for r in reject
]

# Ordina dal risultato più netto a quello meno netto
df_sorted = df.sort_values(by="p-value_corretto").reset_index(drop=True)

print("--- RISULTATI TEST STATISTICO CHI-QUADRO ---")
print(
    df_sorted[
        [
            "UPOS",
            "Polso_assolute",
            "Gen_assolute",
            "p-value_corretto",
            "Significativo (FDR < 0.05)",
        ]
    ]
)