import os
from collections import Counter
from conllu import parse_incr
from tqdm import tqdm
import pandas as pd

# ======================================================
# PARAMETRI
# ======================================================

ROOT_FOLDER = "conllu_files"
OUTPUT_CSV = "ordine_SN.csv"

# ======================================================
# RELAZIONI CONSIDERATE MODIFICATORI DEL NOME
# ======================================================

MODIFIERS = {
    "det",
    "amod",
    "nummod",
    "nmod",
    "acl",
    "acl:relcl",
    "appos",
    "compound"
}

# ======================================================

def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".conllu"):
                yield os.path.join(root, f)

# ======================================================

def analyze_folder(folder_path, folder_name):

    counts = Counter()

    is_isdt = folder_name.lower() == "isdt"

    for conllu_file in tqdm(iter_conllu_files(folder_path),
                            desc=folder_name):

        try:

            with open(conllu_file,
                      encoding="utf-8",
                      errors="replace") as f:

                for sentence in parse_incr(f):

                    if is_isdt:
                        sent_id = sentence.metadata.get("sent_id", "")
                        if "isst_tanl" not in sent_id:
                            continue

                    # figli di ogni token
                    children = {}

                    for tok in sentence:

                        if not isinstance(tok["id"], int):
                            continue

                        head = tok["head"]

                        if isinstance(head, int):
                            children.setdefault(head, []).append(tok)

                    # ricerca delle teste
                    for head in sentence:

                        if not isinstance(head["id"], int):
                            continue

                        upos = head["upos"]

                        if upos not in {"NOUN", "PROPN", "PRON"}:
                            continue

                        mods_before = False
                        mods_after = False

                        for child in children.get(head["id"], []):

                            if child["deprel"] not in MODIFIERS:
                                continue

                            if child["id"] < head["id"]:
                                mods_before = True

                            elif child["id"] > head["id"]:
                                mods_after = True

                        if upos in {"NOUN", "PROPN"}:
                            prefix = "NOUN"

                        else:
                            prefix = "PRON"

                        if mods_before and mods_after:
                            counts[prefix + "_ModNMod"] += 1

                        elif mods_before:
                            counts[prefix + "_ModN"] += 1

                        elif mods_after:
                            counts[prefix + "_NMod"] += 1

        except Exception as e:
            print(conllu_file, e)

    return {
        "cartella": folder_name,

        "NOUN_ModN": counts["NOUN_ModN"],
        "NOUN_NMod": counts["NOUN_NMod"],
        "NOUN_ModNMod": counts["NOUN_ModNMod"],

        "PRON_ModN": counts["PRON_ModN"],
        "PRON_NMod": counts["PRON_NMod"],
        "PRON_ModNMod": counts["PRON_ModNMod"],
    }

# ======================================================
# MAIN
# ======================================================

results = []

for folder in sorted(next(os.walk(ROOT_FOLDER))[1]):

    path = os.path.join(ROOT_FOLDER, folder)

    results.append(analyze_folder(path, folder))

df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV,
          index=False,
          encoding="utf-8")

print(df)

print(f"\nCSV salvato in {OUTPUT_CSV}")