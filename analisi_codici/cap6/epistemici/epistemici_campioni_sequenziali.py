# =========================================================
# ANALISI MARCATORI - AUTOMATICO (VERSIONE FORME + REGEX)
# =========================================================

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from conllu import parse_incr
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations

# -------------------------
# PARAMETRI
# -------------------------
CONLLU_ROOT = "conllu_files"
MARKERS_CSV = "lista_epistemici.csv"
SAMPLE_SIZE = 1000
AGG_COLS = ["certezza", "polarity", "struttura", "prende_complemento"]

# -------------------------
# 1. Carica marcatori
# -------------------------
try:
    # Usiamo sep=';' e engine='python' per massima compatibilità
    markers_df = pd.read_csv(MARKERS_CSV, engine='python')
    markers_df.columns = markers_df.columns.str.strip()
    markers_df = markers_df.fillna("")
    
    # Pulizia: tutto in minuscolo per il confronto
    markers_df["forme"] = markers_df["forme"].str.lower().str.strip()

    # Dizionario: forma -> categorie (polarity, struttura, etc.)
    marker_dict = {}
    for _, row in markers_df.iterrows():
        if row["forme"]:
            marker_dict[row["forme"]] = {col: row[col] for col in AGG_COLS}
            
    print(f"✔ Caricati {len(marker_dict)} marcatori da {MARKERS_CSV}")
except Exception as e:
    print(f"✘ Errore nel caricamento del CSV: {e}")
    exit()

# -------------------------
# 2. Utilità file
# -------------------------
def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

# -------------------------
# 3. Leggi FORME (non lemmi)
# -------------------------
def read_tokens_from_folder(folder):
    """
    Legge le FORME superficiali dai file .conllu.
    - Esclude token PUNCT
    - Filtra ISDT se necessario
    """
    corpus_name = os.path.basename(folder).lower()
    tokens = []

    for conllu_file in sorted(iter_conllu_files(folder)):
        try:
            with open(conllu_file, "r", encoding="utf-8", errors="replace") as f:
                for sentence in parse_incr(f):
                    # Eventuale filtro specifico per corpus ISDT
                    if corpus_name == "isdt":
                        sent_id = sentence.metadata.get("sent_id", "")
                        if "isst_tanl" not in sent_id:
                            continue

                    for token in sentence:
                        # --- PRENDIAMO LA FORMA (form) ---
                        form = token.get("form")
                        upos = token.get("upostag") or token.get("upos") or ""
                        
                        # Scartiamo punteggiatura e token vuoti
                        if form and upos != "PUNCT":
                            tokens.append(form.lower())
        except Exception as e:
            print(f"✘ Errore file {conllu_file}: {e}")
    return tokens

# -------------------------
# 4. Campionamento
# -------------------------
def make_sequential_samples(tokens, sample_size):
    if len(tokens) <= sample_size:
        return [tokens]
    samples = []
    for i in range(0, len(tokens), sample_size):
        chunk = tokens[i : i + sample_size]
        if len(chunk) == sample_size:
            samples.append(chunk)
    return samples

# -------------------------
# 5. Conteggio Marcatori (REGEX - Parola Intera)
# -------------------------
def count_markers_in_sample(tokens):
    total = 0
    agg_counts = {col: Counter() for col in AGG_COLS}
    # Uniamo i token con uno spazio per permettere il match di multi-words
    text = " ".join(tokens)
    
    for form, cats in marker_dict.items():
        # \b assicura che il match avvenga solo su parola intera
        pattern = r'\b' + re.escape(form) + r'\b'
        matches = re.findall(pattern, text)
        occ = len(matches)
        
        if occ > 0:
            total += occ
            for col in AGG_COLS:
                valore = cats[col] if cats[col] else "NA"
                agg_counts[col][valore] += occ
                
    return total, agg_counts

# -------------------------
# 6. Esecuzione Analisi
# -------------------------
all_rows = []
if not os.path.exists(CONLLU_ROOT):
    print(f"✘ Errore: la cartella {CONLLU_ROOT} non esiste!")
    exit()

for subfolder in sorted(os.listdir(CONLLU_ROOT)):
    folder_path = os.path.join(CONLLU_ROOT, subfolder)
    if os.path.isdir(folder_path):
        print(f"➤ Processando corpus: {subfolder}...")
        tokens = read_tokens_from_folder(folder_path)
        samples = make_sequential_samples(tokens, SAMPLE_SIZE)
        
        for i, sample in enumerate(samples):
            total, agg_counts = count_markers_in_sample(sample)
            row = {
                "corpus": subfolder, 
                "sample_id": i, 
                "markers": total, 
                "tokens": len(sample)
            }
            # Appiattimento dei conteggi per polarità, struttura, etc.
            for col in AGG_COLS:
                for key, val in agg_counts[col].items():
                    row[f"{col}_{key}"] = val
            all_rows.append(row)

# Salvataggio dati grezzi
df = pd.DataFrame(all_rows).fillna(0)
df.to_csv("all_samples.csv", index=False, encoding="utf-8")
print("\n✔ Salvato: all_samples.csv")

# -------------------------
# 7. Statistiche e Modelli
# -------------------------
if not df.empty:
    # Mann-Whitney per differenze tra corpora
    corpora = df['corpus'].unique()
    mw_results = []
    for c1, c2 in combinations(corpora, 2):
        a = df[df['corpus'] == c1]['markers']
        b = df[df['corpus'] == c2]['markers']
        if len(a) > 0 and len(b) > 0:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            mw_results.append({"corpus1": c1, "corpus2": c2, "U": u, "p_value": p})
    
    pd.DataFrame(mw_results).to_csv("mannwhitney_results.csv", index=False)
    print("✔ Salvato: mannwhitney_results.csv")

    # Regressione Negativa Binomiale (IRR)
    try:
        df["log_tokens"] = np.log(df["tokens"])
        model = smf.glm(
            formula="markers ~ corpus",
            data=df,
            family=sm.families.NegativeBinomial(),
            offset=df["log_tokens"]
        ).fit()
        
        res_df = pd.DataFrame({
            "coef": model.params,
            "p_value": model.pvalues,
            "IRR": np.exp(model.params)
        })
        res_df.to_csv("glm_results.csv")
        print("✔ Salvato: glm_results.csv")
        print("\nSintesi Modello:")
        print(model.summary())
    except Exception as e:
        print(f"⚠ Errore nel calcolo del GLM: {e}")

else:
    print("⚠ Nessun dato trovato. Verifica i file CoNLL-U.")