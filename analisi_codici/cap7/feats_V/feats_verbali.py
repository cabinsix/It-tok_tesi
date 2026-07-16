import os
import csv
from collections import defaultdict

# =========================
# CONFIGURAZIONE
# =========================

BASE_DIR = "conllu_files"
OUTPUT_DIR = "csv_feats_verbali"

MAX_AUX_DEPTH = 3

EXCLUDED_CORPORA = {
    "polso",
    "gen",
    "markit",
    "Ittok",
}

# CoNLL-U base (senza FEATS)
BASE_FIELDS = [
    "id", "form", "lemma", "upos", "xpos",
    "head", "deprel", "deps", "misc"
]

# raccolta globale FEATS
ALL_FEATS = set()

# riepilogo globale feats
feat_summary = defaultdict(int)

# =========================
# PARSING CONLLU
# =========================

def parse_feats(feats_str):
    feats = {}
    if feats_str in ("_", "", None):
        return feats
    for fv in feats_str.split("|"):
        if "=" in fv:
            f, v = fv.split("=", 1)
            feats[f] = v
            ALL_FEATS.add(f)
    return feats


def parse_conllu(path):
    sentences = []
    current = {"sent_id": None, "tokens": []}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if not line:
                if current["tokens"]:
                    sentences.append(current)
                current = {"sent_id": None, "tokens": []}
                continue

            if line.startswith("# sent_id"):
                current["sent_id"] = line.split("=", 1)[1].strip()
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue

            feats = parse_feats(fields[5])

            token = {
                "id": int(fields[0]),
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "xpos": fields[4],
                "feats": feats,
                "head": fields[6],
                "deprel": fields[7],
                "deps": fields[8],
                "misc": fields[9],
            }
            current["tokens"].append(token)

    return sentences

# =========================
# INDICI
# =========================

def build_index(tokens):
    by_head = defaultdict(list)
    for t in tokens:
        head = int(t["head"]) if t["head"].isdigit() else None
        by_head[head].append(t)
    return by_head

# =========================
# AUX CHAIN
# =========================

def find_aux_chain(verb_id, by_head):
    chain = []
    current = verb_id

    for _ in range(MAX_AUX_DEPTH):
        found = None
        for t in by_head.get(current, []):
            if t["upos"] == "AUX":
                found = t
                break
        if not found:
            break
        chain.append(found)
        current = found["id"]

    return chain

# =========================
# FEATS SUMMARY
# =========================

def update_feat_summary(token, corpus):
    for feat, val in token["feats"].items():
        feat_summary[(corpus, token["upos"], feat, val)] += 1

# =========================
# PROCESS CORPUS
# =========================

def process_corpus(corpus_path, corpus_name, rows):
    for file in os.listdir(corpus_path):
        if not file.endswith(".conllu"):
            continue

        sentences = parse_conllu(os.path.join(corpus_path, file))

        for sent in sentences:

            if corpus_name == "isdt":
                if sent["sent_id"] is None or "isst_tanl" not in sent["sent_id"]:
                    continue

            by_head = build_index(sent["tokens"])

            for token in sent["tokens"]:
                if token["upos"] != "VERB":
                    continue

                aux_chain = find_aux_chain(token["id"], by_head)

                update_feat_summary(token, corpus_name)
                for aux in aux_chain:
                    update_feat_summary(aux, corpus_name)

                row = {
                    "corpus": corpus_name,
                    "conllu_file": file,
                    "sent_id": sent["sent_id"],
                }

                # VERB base
                for f in BASE_FIELDS:
                    row[f"verb_{f}"] = token[f]

                # VERB feats
                for feat in ALL_FEATS:
                    row[f"verb_{feat}"] = token["feats"].get(feat, "")

                # AUX
                for i in range(MAX_AUX_DEPTH):
                    prefix = f"aux{i+1}_"
                    if i < len(aux_chain):
                        aux = aux_chain[i]
                        for f in BASE_FIELDS:
                            row[prefix + f] = aux[f]
                        for feat in ALL_FEATS:
                            row[prefix + feat] = aux["feats"].get(feat, "")
                    else:
                        for f in BASE_FIELDS:
                            row[prefix + f] = ""
                        for feat in ALL_FEATS:
                            row[prefix + feat] = ""

                rows.append(row)

# =========================
# WRITE CSV PER CORPUS
# =========================

def write_corpus_csv(corpus, rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{corpus}.csv")

    fieldnames = (
        ["corpus", "conllu_file", "sent_id"]
        + [f"verb_{f}" for f in BASE_FIELDS]
        + [f"verb_{feat}" for feat in sorted(ALL_FEATS)]
        + [
            f"aux{i}_{f}"
            for i in range(1, MAX_AUX_DEPTH + 1)
            for f in BASE_FIELDS
        ]
        + [
            f"aux{i}_{feat}"
            for i in range(1, MAX_AUX_DEPTH + 1)
            for feat in sorted(ALL_FEATS)
        ]
    )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# =========================
# FEATS SUMMARY CSV
# =========================

def write_feat_summary():
    out_path = os.path.join(OUTPUT_DIR, "feats_summary_all_corpora.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corpus", "pos", "feat", "value", "count"])
        for (c, p, f, v), cnt in sorted(feat_summary.items()):
            writer.writerow([c, p, f, v, cnt])

# =========================
# MAIN
# =========================

def main():
    corpus_rows = defaultdict(list)

    for corpus in os.listdir(BASE_DIR):
        if corpus in EXCLUDED_CORPORA:
            continue

        path = os.path.join(BASE_DIR, corpus)
        if os.path.isdir(path):
            process_corpus(path, corpus, corpus_rows[corpus])

    for corpus, rows in corpus_rows.items():
        write_corpus_csv(corpus, rows)

    write_feat_summary()

if __name__ == "__main__":
    main()
