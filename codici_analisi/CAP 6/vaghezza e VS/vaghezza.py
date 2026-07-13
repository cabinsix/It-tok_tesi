import os
import csv
from collections import defaultdict
from conllu import parse
from lxml import etree as ET

# =========================
# CONFIGURAZIONE
# =========================
base_path = "conllu_files"
wordnet_file = "IWN-OMW_LMF_v1.0.xml"
output_csv = "vaghezza_per_corpus.csv"

# POS ItalWordNet
pos_list = ["n", "v", "a", "r"]

# Mapping UD → ItalWordNet
ud2iwn = {
    "noun": "n",
    "verb": "v",
    "adj": "a",
    "adv": "r"
}

# =========================
# PARSING XML WORDNET
# =========================
print("📖 Parsing ItalWordNet XML (recover mode)...")

parser = ET.XMLParser(
    recover=True,
    huge_tree=True,
    remove_blank_text=True
)

tree = ET.parse(wordnet_file, parser)
root = tree.getroot()

print("✅ XML caricato con recover mode")

print("LexicalEntry:", len(root.findall(".//LexicalEntry")))
print("Synset:", len(root.findall(".//Synset")))

lemma_synsets = defaultdict(list)
synset_rel = {}

# --- LexicalEntry ---
for lex in root.findall(".//LexicalEntry"):
    lemma_elem = lex.find("Lemma")
    if lemma_elem is None:
        continue

    lemma = (lemma_elem.get("writtenForm") or "").lower()
    pos = (lemma_elem.get("partOfSpeech") or "").lower()

    for sense in lex.findall("Sense"):
        syn = sense.get("synset")
        if syn:
            lemma_synsets[(lemma, pos)].append(syn)

# --- Synset ---
for syn in root.findall(".//Synset"):
    sid = syn.get("id")
    rels = []
    for r in syn.findall("SynsetRelation"):
        rt = r.get("relType")
        tgt = r.get("target")
        rels.append((rt, tgt))
    synset_rel[sid] = rels

# =========================
# PROFONDITA' SYNSET
# =========================
depth_cache = {}

def compute_depth(syn):
    if syn in depth_cache:
        return depth_cache[syn]

    rels = synset_rel.get(syn, [])
    hypers = [t for (rt, t) in rels if rt == "hypernym"]

    if not hypers:
        depth_cache[syn] = 0
    else:
        depth_cache[syn] = 1 + min(
            compute_depth(h) for h in hypers if h in synset_rel
        )

    return depth_cache[syn]

for s in synset_rel.keys():
    compute_depth(s)

depth_max = max(depth_cache.values()) if depth_cache else 1

# =========================
# CALCOLO P(w), D(w), V(w)
# =========================
sense_counts = {
    lp: len(syns) for lp, syns in lemma_synsets.items()
}
max_senses = max(sense_counts.values()) if sense_counts else 1

lemma_measures = {}

for (lemma, pos), syns in lemma_synsets.items():

    if pos not in pos_list:
        continue

    p_norm = (len(syns) - 1) / max_senses if max_senses > 1 else 0

    depths = [depth_cache.get(s, 0) for s in syns]
    avg_depth = sum(depths) / len(depths) if depths else 0

    d_norm = 1 - (avg_depth / depth_max if depth_max > 0 else 0)

    vag = p_norm * d_norm

    lemma_measures[(lemma, pos)] = {
        "P": round(p_norm, 4),
        "D": round(d_norm, 4),
        "V": round(vag, 4)
    }

print("📊 Lemmi con misure:", len(lemma_measures))

# =========================
# CARICAMENTO CORPORA
# =========================
dataset = {}

for corpus in os.listdir(base_path):

    path = os.path.join(base_path, corpus)

    if not os.path.isdir(path):
        continue

    sentences = []

    for file in os.listdir(path):

        if not file.endswith(".conllu"):
            continue


        file_path = os.path.join(path, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        for s in parse(content):

    # recupera metadata in minuscolo per sicurezza
            meta = {k.lower(): v.lower() for k,v in s.metadata.items()}

    # prendi il sent_id
            sent_id = meta.get("sent_id","")

    # --- filtro solo ISST dentro ISDT ---
            if "isdt" in corpus.lower():
                if "isst" not in sent_id:
                    continue

            sentences.append(s)



    dataset[corpus] = sentences

print("📚 Corpora caricati:", len(dataset))

# =========================
# ESTRAZIONE TOKEN
# =========================
output_rows = []

for corpus, sents in dataset.items():

    for s in sents:

        for t in s:

            lemma = (t["lemma"] or "").lower()
            upos = (t["upos"] or "").lower()

            pos = ud2iwn.get(upos)
            if pos is None:
                continue

            key = (lemma, pos)

            if key not in lemma_measures:
                continue

            measures = lemma_measures[key]

            output_rows.append({
                "Corpus": corpus,
                "Lemma": lemma,
                "POS": pos,
                "P_w": measures["P"],
                "D_w": measures["D"],
                "V_w": measures["V"]
            })

print("🧾 Token validi:", len(output_rows))

# =========================
# SALVATAGGIO CSV
# =========================
fieldnames = ["Corpus", "POS", "Lemma", "P_w", "D_w", "V_w"]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"✅ Risultati salvati in {output_csv}")
