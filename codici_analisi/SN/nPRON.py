import os
from collections import Counter
from conllu import parse_incr
from tqdm import tqdm
import pandas as pd

ROOT_FOLDER = "conllu_files"
OUTPUT_CSV = "conteggio_sn.csv"

# ======================================================
# ITERATORE FILE
# ======================================================

def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

# ======================================================
# CONTEGGIO SN
# ======================================================

def count_nps(sub_path, subfolder):

    counts = Counter()

    is_isdt = subfolder.lower() == "isdt"

    for conllu_file in tqdm(iter_conllu_files(sub_path),
                            desc=subfolder,
                            unit="file"):

        try:
            with open(conllu_file, encoding="utf-8", errors="replace") as f:

                for sentence in parse_incr(f):

                    if is_isdt:
                        sent_id = sentence.metadata.get("sent_id", "")
                        if "isst_tanl" not in sent_id:
                            continue

                    for token in sentence:

                        if not isinstance(token["id"], int):
                            continue

                        upos = token["upos"]

                        # testa nominale
                        if upos == "NOUN":
                            counts["SN_testa_nominale"] += 1

                        # se vuoi includere anche i nomi propri
                        elif upos == "PROPN":
                            counts["SN_testa_nominale"] += 1

                        # testa pronominale
                        elif upos == "PRON":
                            counts["SN_testa_pronominale"] += 1

        except Exception as e:
            print(conllu_file, e)

    totale = (counts["SN_testa_nominale"] +
              counts["SN_testa_pronominale"])

    return {
        "cartella": subfolder,
        "SN_testa_nominale": counts["SN_testa_nominale"],
        "SN_testa_pronominale": counts["SN_testa_pronominale"],
        "Totale_SN": totale
    }

# ======================================================
# MAIN
# ======================================================

walk = next(os.walk(ROOT_FOLDER), (None, [], []))
subfolders = walk[1]

results = []

for subfolder in sorted(subfolders):
    sub_path = os.path.join(ROOT_FOLDER, subfolder)
    results.append(count_nps(sub_path, subfolder))

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(df)
print(f"\nCSV salvato in {OUTPUT_CSV}")