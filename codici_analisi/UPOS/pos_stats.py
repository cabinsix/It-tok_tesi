from collections import Counter
import os
from pathlib import Path
import re
import sys
from conllu import parse_incr
import pandas as pd

# Punto di partenza principale
BASE_DIR = "conllu_files"


def get_subfolders(base_path):
    """Restituisce l'elenco delle sotto-cartelle dirette (es.

    isdt, gen, ecc.).
    """
    return [
        f.name
        for f in Path(base_path).iterdir()
        if f.is_dir() and not f.name.startswith(".")
    ]


def iter_conllu_files(folder_path):
    """Genera tutti i file .conllu ricorsivamente per una specifica cartella."""
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".conllu") and not f.startswith("."):
                yield os.path.join(root, f)


def read_sentences_from_folder(folder_path, subfolder_name):
    """Legge le frasi da una specifica cartella, applicando filtri speciali (es.

    per ISDT).
    """
    sentences = []
    is_isdt = subfolder_name.lower() == "isdt"

    for file in iter_conllu_files(folder_path):
        print(f"[{subfolder_name.upper()}] Leggo: {file}")
        try:
            with open(file, "r", encoding="utf-8", errors="replace") as f:
                for sentence in parse_incr(f):
                    # Se siamo in ISDT, filtriamo per sent_id
                    if is_isdt:
                        sent_id = sentence.metadata.get("sent_id", "")
                        if "isst_tanl" not in sent_id:
                            continue

                    sentences.append(sentence)
        except Exception as e:
            print(f"ERRORE nel file: {file} -> {e}")
    return sentences


def salva_counter(counter, nomefile):
    data = counter.most_common()
    df = pd.DataFrame(data, columns=["item", "freq"])
    df.to_csv(nomefile, index=False, encoding="utf-8")


# ==========================================
# CORE LOOP: Processa ogni sotto-cartella
# ==========================================

# 1. Crea la cartella "output_pos" nella cartella in cui operi (cartella corrente)
output_dir = os.path.join(os.getcwd(), "output_pos")
os.makedirs(output_dir, exist_ok=True)

subfolders = get_subfolders(BASE_DIR)
print(f"Cartelle trovate da elaborare: {subfolders}")
print(f"Tutti gli output saranno salvati in: {output_dir}\n")

for subfolder in subfolders:
    folder_path = os.path.join(BASE_DIR, subfolder)
    print(
        f"=========================================\nINIZIO ELABORAZIONE CARTELLA: {subfolder.upper()}\n========================================="
    )

    # Leggi le frasi filtrate
    sentences = read_sentences_from_folder(folder_path, subfolder)

    # Inizializzazione contatori
    total_tokens = 0
    tokens_no_punct = 0

    freq = Counter()
    upos_freq = Counter()
    deps = Counter()
    feat_counter = Counter()
    lemma_freq = Counter()
    comb_freq = Counter()
    comb_forme_freq = Counter()
    dm_forme_freq = Counter()
    dm_freq = Counter()

    cconj_lemma_freq = Counter()
    sconj_lemma_freq = Counter()
    pron_lemma_freq = Counter()
    noun_lemma_freq = Counter()
    verb_lemma_freq = Counter()

    # Estrazione informazioni
    for sent in sentences:
        for tok in sent:
            if isinstance(tok["id"], int):
                deps[tok["deprel"]] += 1

        for tok in sent:
            if isinstance(tok["id"], int):
                total_tokens += 1

                feats = tok.get("feats") or {}
                for f, v in feats.items():
                    feat_counter[f"{f}={v}"] += 1

                if tok["lemma"]:
                    lemma_freq[tok["lemma"].lower()] += 1

                if tok["upos"] != "PUNCT":
                    tokens_no_punct += 1

                    form_low = tok["form"].lower() if tok["form"] else ""
                    lemma_low = tok["lemma"].lower() if tok["lemma"] else ""
                    upos = tok["upos"]

                    freq[form_low] += 1
                    upos_freq[upos] += 1

                    if lemma_low:
                        comb_freq[f"{lemma_low}_{upos}"] += 1
                    if form_low:
                        comb_forme_freq[f"{form_low}_{upos}"] += 1

                    if upos == "CCONJ" and lemma_low:
                        cconj_lemma_freq[lemma_low] += 1
                    elif upos == "SCONJ" and lemma_low:
                        sconj_lemma_freq[lemma_low] += 1
                    elif upos == "PRON" and lemma_low:
                        pron_lemma_freq[lemma_low] += 1
                    elif upos == "NOUN" and lemma_low:
                        noun_lemma_freq[lemma_low] += 1
                    elif upos == "VERB" and lemma_low:
                        verb_lemma_freq[lemma_low] += 1

                misc = tok.get("misc", None)
                is_dm = False
                if isinstance(misc, dict) and "DM" in misc:
                    is_dm = True
                elif isinstance(misc, str) and "DM" in misc.split("|"):
                    is_dm = True

                if is_dm and tok["form"]:
                    form_low = tok["form"].lower()
                    dm_freq[form_low] += 1
                    if tok["upos"]:
                        dm_forme_freq[f"{form_low}_{tok['upos']}"] += 1

    print(f"\n[{subfolder.upper()}] Risultati rapidi:")
    print("Numero totale di token:", total_tokens)
    print("Token diversi da PUNCT:", tokens_no_punct)

    # prefisso dinamico basato sul nome della cartella (es. "gen_", "isdt_")
    pfx = f"{subfolder}_"

    # Salvataggio dei file CSV con il prefisso del corpus dentro 'output_pos'
    salva_counter(freq, os.path.join(output_dir, f"{pfx}01_frequenze_forme.csv"))
    salva_counter(
        deps, os.path.join(output_dir, f"{pfx}02_frequenze_dipendenze.csv")
    )
    salva_counter(feat_counter, os.path.join(output_dir, f"{pfx}03_feature.csv"))
    salva_counter(lemma_freq, os.path.join(output_dir, f"{pfx}04_lemmi.csv"))
    salva_counter(comb_freq, os.path.join(output_dir, f"{pfx}05_lemmi_UPOS.csv"))
    salva_counter(
        comb_forme_freq, os.path.join(output_dir, f"{pfx}06_forme_UPOS.csv")
    )
    salva_counter(
        dm_forme_freq, os.path.join(output_dir, f"{pfx}07_DM_UPOS.csv")
    )
    salva_counter(dm_freq, os.path.join(output_dir, f"{pfx}08_DM_forme.csv"))
    salva_counter(
        cconj_lemma_freq, os.path.join(output_dir, f"{pfx}09_CCONJ_lemmi.csv")
    )
    salva_counter(
        sconj_lemma_freq, os.path.join(output_dir, f"{pfx}10_SCONJ_lemmi.csv")
    )
    salva_counter(
        pron_lemma_freq, os.path.join(output_dir, f"{pfx}11_PRON_lemmi.csv")
    )
    salva_counter(
        noun_lemma_freq, os.path.join(output_dir, f"{pfx}12_NOUN_lemmi.csv")
    )
    salva_counter(
        verb_lemma_freq, os.path.join(output_dir, f"{pfx}13_VERB_lemmi.csv")
    )
    salva_counter(
        upos_freq, os.path.join(output_dir, f"{pfx}00_UPOS_frequenze.csv")
    )

    # Creazione della matrice pivot
    all_counters = {
        "freq": freq,
        "upos": upos_freq,
        "deps": deps,
        "feat": feat_counter,
        "lemma": lemma_freq,
        "lemma_upos": comb_freq,
        "forme_upos": comb_forme_freq,
        "dm_upos": dm_forme_freq,
        "dm": dm_freq,
        "cconj_lemma": cconj_lemma_freq,
        "sconj_lemma": sconj_lemma_freq,
        "pron_lemma": pron_lemma_freq,
        "noun_lemma": noun_lemma_freq,
        "verb_lemma": verb_lemma_freq,
    }

    df_pivot = pd.DataFrame.from_dict(
        {name: dict(counter) for name, counter in all_counters.items()},
        orient="index",
    ).T
    df_pivot.to_csv(
        os.path.join(output_dir, f"output_matrice_{subfolder}.csv"),
        encoding="utf-8",
    )

    print(f"Salvataggio completato per la cartella: {subfolder}\n")

print("\n--- PROCESSO TERMINATO CON SUCCESSO ---")