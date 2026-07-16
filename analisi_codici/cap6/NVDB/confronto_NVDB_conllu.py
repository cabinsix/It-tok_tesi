import os
import pandas as pd
from collections import Counter
from conllu import parse_incr
from tqdm import tqdm

# ======================================================
# PERCORSI
# ======================================================
root_folder = "conllu_files"  # cambiare con il percorso reale
txt_file = "nvdb.full.txt"
output_folder = "conllu_based_nvdb"
os.makedirs(output_folder, exist_ok=True)

# ======================================================
# CARICAMENTO TXT NVDB
# ======================================================
nvdb_set = set()
with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if line:
            nvdb_set.add(line.split()[0].lower())

print(f"Caricate {len(nvdb_set)} parole dal TXT NVDB")

# ======================================================
# FUNZIONE PER ITERARE FILE CONLLU
# ======================================================
def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

# ======================================================
# FUNZIONE SPECIFICA PER ISDT
# ======================================================
def read_all_sentences_isdt(path):
    """
    Legge SOLO le frasi ISDT con sent_id contenente 'isst_tanl'
    """
    for file in iter_conllu_files(path):
        print("Leggo:", file)
        try:
            with open(file, "r", encoding="utf-8", errors="replace") as f:
                for sentence in parse_incr(f):
                    sent_id = sentence.metadata.get("sent_id", "")
                    if "isst_tanl" in sent_id:
                        yield file, sentence
        except Exception as e:
            print("ERRORE nel file:", file)
            raise e

# ======================================================
# PROCESSO SOTTOCARTELLE
# ======================================================
walk_result = next(os.walk(root_folder), (None, [], []))
all_subfolders = walk_result[1]  # lista sottocartelle

if not all_subfolders:
    print(f"Nessuna sottocartella trovata in {root_folder}")
else:
    for subfolder in sorted(all_subfolders):
        sub_path = os.path.join(root_folder, subfolder)
        lemma_counter = Counter()

        print(f"\nProcessando sottocartella: {subfolder}")

        if subfolder.lower() == "isdt":
            # usa la funzione speciale ISDT
            for file, sentence in tqdm(read_all_sentences_isdt(sub_path), desc="Lettura frasi ISDT", unit="frase"):
                for token in sentence:
                    upos = token.get("upostag") or token.get("upos") or ""
                    lemma = token.get("lemma")
                    if not lemma:
                        continue
                    lemma = lemma.lower()
                    # AGGIUNTO "NUM" per escludere i numeri
                    if upos in ("PROPN", "PUNCT", "NUM"):
                        continue
                    lemma_counter[lemma] += 1
        else:
            # comportamento standard per tutte le altre sottocartelle
            for conllu_file in tqdm(iter_conllu_files(sub_path), desc=f"Lettura {subfolder}", unit="file"):
                try:
                    with open(conllu_file, "r", encoding="utf-8", errors="replace") as f:
                        for sentence in parse_incr(f):
                            for token in sentence:
                                upos = token.get("upostag") or token.get("upos") or ""
                                lemma = token.get("lemma")
                                if not lemma:
                                    continue
                                lemma = lemma.lower()
                                # AGGIUNTO "NUM" per escludere i numeri
                                if upos in ("PROPN", "PUNCT", "NUM"):
                                    continue
                                lemma_counter[lemma] += 1
                except Exception as e:
                    print(f"✘ Errore file {conllu_file}: {e}")

        # crea DataFrame
        df = pd.DataFrame(
            [(lemma, freq, "VB" if lemma in nvdb_set else "") for lemma, freq in lemma_counter.items()],
            columns=["lemma", "freq", "VB"]
        ).sort_values(by="freq", ascending=False)

        # salva CSV
        output_csv = os.path.join(output_folder, f"{subfolder}.csv")
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"✔ CSV salvato per sottocartella {subfolder}: {output_csv}")