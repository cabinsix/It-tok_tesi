import os
import pandas as pd
from conllu import parse_incr
from collections import Counter
from random import shuffle
from scipy.stats import chi2_contingency


# -------------------------
# Percorsi e parametri
# -------------------------
root_folder = "conllu_files"
output_csv = "ordine_costituenti.csv"
SAMPLE_SIZE = 5000  # token
output_test_csv = "ordine_costituenti_chi2.csv"  # CSV dei test chi-quadro

# corpora da escludere
exclude_corpora = {"polso", "gen", "Ittok", "postwita","markit"}

# -------------------------
# Funzioni ausiliarie
# -------------------------
def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

def read_sentences(corpus_name, path):
    """
    Legge le frasi da un corpus
    Per ISDT, considera solo sent_id contenente 'isst_tanl'
    """
    for file in iter_conllu_files(path):
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            for sentence in parse_incr(f):
                sent_id = sentence.metadata.get("sent_id", "")
                if corpus_name.lower() == "isdt" and "isst_tanl" not in sent_id:
                    continue
                yield sentence

# -------------------------
# Analisi ordine costituenti
# -------------------------
# -------------------------
# Analisi ordine costituenti per POS
# -------------------------
results = []

for corpus_name in os.listdir(root_folder):
    if corpus_name in exclude_corpora:
        continue
    corpus_path = os.path.join(root_folder, corpus_name)
    if not os.path.isdir(corpus_path):
        continue

    # raccoglie tutti i token validi (per eventuale campionamento)
    tokens = []
    sentences = list(read_sentences(corpus_name, corpus_path))
    for sent in sentences:
        for tok in sent:
            if tok['upostag'] == "PUNCT":
                continue
            tokens.append(tok)
    if len(tokens) > SAMPLE_SIZE:
        shuffle(tokens)
        tokens = tokens[:SAMPLE_SIZE]

    # -------------------------
    # dizionari per soggetto e oggetto per POS
    # -------------------------
    subj_counts = {}  # es: {"NOUN": {"SV": 10, "VS":2, "total":12}, ...}
    obj_counts = {}   # es: {"NOUN": {"OV": 8, "VO":3, "total":11}, ...}

    for sent in sentences:
        id2pos = {tok['id']: idx for idx, tok in enumerate(sent)}
        for tok in sent:
            head = tok['head']
            dep = tok['deprel']
            pos = tok['upostag']  # POS del token
            if dep in ("nsubj", "csubj"):
                if pos not in subj_counts:
                    subj_counts[pos] = {"SV":0, "VS":0, "total":0}
                subj_counts[pos]["total"] += 1
                if head in id2pos:
                    if id2pos[tok['id']] < id2pos[head]:
                        subj_counts[pos]["SV"] += 1
                    else:
                        subj_counts[pos]["VS"] += 1
            elif dep in ("obj", "ccomp"):
                if pos not in obj_counts:
                    obj_counts[pos] = {"OV":0, "VO":0, "total":0}
                obj_counts[pos]["total"] += 1
                if head in id2pos:
                    if id2pos[tok['id']] < id2pos[head]:
                        obj_counts[pos]["OV"] += 1
                    else:
                        obj_counts[pos]["VO"] += 1

    # -------------------------
    # Salva risultati per corpus in formato "long" (POS per riga)
    # -------------------------
    for pos, counts in subj_counts.items():
        results.append({
            "corpus": corpus_name,
            "tipo": "soggetto",
            "POS": pos,
            "total": counts["total"],
            "SV": counts["SV"],
            "VS": counts["VS"]
        })
    for pos, counts in obj_counts.items():
        results.append({
            "corpus": corpus_name,
            "tipo": "oggetto",
            "POS": pos,
            "total": counts["total"],
            "OV": counts["OV"],
            "VO": counts["VO"]
        })

# -------------------------
# Salva CSV
# -------------------------
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✔ Risultati salvati in {output_csv}")
