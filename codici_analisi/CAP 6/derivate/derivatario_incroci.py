import os
import pandas as pd
from collections import Counter
from conllu import parse_incr

# ======================================================
# CONFIG
# ======================================================
ROOT_CONLLU = "conllu_files"
DERIVATARIO = "derivatario.csv"
OUTPUT = "output_complesse_derivatario"

os.makedirs(OUTPUT, exist_ok=True)

POS_MAP = {"NOUN":"N", "VERB":"V", "ADJ":"J", "ADV":"R"}
POS_SET = {"N","V","J","R"}

# ======================================================
# TOTALI CORPORA (forniti da te)
# ======================================================
TOTALI = {
    "gen": {"lemmi":3759, "token":35252},
    "kiparla": {"lemmi":1240, "token":9348},
    "polso": {"lemmi":4247, "token":33031},
    "isdt": {"lemmi":9357, "token":69390},
    "Ittok": {"lemmi":6294, "token":68283},
    "postwita": {"lemmi":13392, "token":124420},
    "trittok_polso": {"lemmi":1119, "token":5321},
    "trittok_gen": {"lemmi":917, "token":4411},
    "tr-it-tok": {"lemmi":1657, "token":9732}
}

# ======================================================
# DERIVATARIO
# ======================================================

DERIVATARIO = "derivatario.csv"   # o .txt

df_der = pd.read_csv(
    DERIVATARIO,
    sep=";",
    header=None,
    names=[
        "id",
        "DERIVATIVE",
        "BASE",
        "wfp1",
        "wfp2",
        "wfp3",
        "wfp4",
        "wfp5",
        "wfp6"
    ],
    dtype=str
).fillna("")

def extract_affixes(row):
    affissi = []
    parasintesi = False

    for col in ["wfp1","wfp2","wfp3","wfp4","wfp5","wfp6"]:
        v = row[col]
        if not v:
            continue

        # controllo parasintesi
        if "-P" in v:
            parasintesi = True

        # ignora root
        if "root" in v:
            continue

        # estrae il nome dell'affisso
        aff = v.split(":")[0]
        if aff:
            affissi.append(aff)

    # REGOLA PRIORITARIA
    if parasintesi:
        return "PARASINTESI", "PARASINTESI"

    if not affissi:
        return "", ""

    # ultimo affisso = principale
    return affissi[-1], "|".join(affissi)


DERIV_INDEX = {}

for _, r in df_der.iterrows():
    lemma = r["DERIVATIVE"].lower()
    affissi = extract_affixes(r)
    if not affissi:
        continue
    DERIV_INDEX.setdefault(lemma, []).append({
        "id": r["id"],
        "base": r["BASE"],
        "affissi_tutti": "|".join(affissi),
        "affisso_principale": affissi[-1],
        **{f"wfp{i}": r[f"wfp{i}"] for i in range(1,7)}
    })

# ======================================================
# CONLLU ITERATORS
# ======================================================
def iter_conllu(path):
    for root,_,files in os.walk(path):
        for f in files:
            if f.endswith(".conllu"):
                yield os.path.join(root,f)

def read_isdt(path):
    for f in iter_conllu(path):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for sent in parse_incr(fh):
                if "isst_tanl" in sent.metadata.get("sent_id",""):
                    yield f, sent

# ======================================================
# ESTRAZIONE
# ======================================================
ROWS = []

for dirpath,_,files in os.walk(ROOT_CONLLU):
    if dirpath == ROOT_CONLLU:
        continue

    corpus = os.path.basename(dirpath)
    if corpus.lower() == "markit":
        continue

    print("PROCESSO:", corpus)

    if corpus.lower() == "isdt":
        iterator = read_isdt(dirpath)
        for f,sent in iterator:
            for t in sent:
                if isinstance(t["id"], tuple):
                    continue
                lemma = t["lemma"].lower()
                if lemma not in DERIV_INDEX:
                    continue
                if t["upos"] not in POS_MAP:
                    continue
                for d in DERIV_INDEX[lemma]:
                    ROWS.append({
                        "corpus": corpus,
                        "token": t["form"].lower(),
                        "lemma": lemma,
                        "upos": POS_MAP[t["upos"]],
                        **d
                    })
    else:
        for f in files:
            if not f.endswith(".conllu"):
                continue
            with open(os.path.join(dirpath,f), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    lemma = parts[2].lower()
                    if lemma not in DERIV_INDEX:
                        continue
                    if parts[3] not in POS_MAP:
                        continue
                    for d in DERIV_INDEX[lemma]:
                        ROWS.append({
                            "corpus": corpus,
                            "token": parts[1].lower(),
                            "lemma": lemma,
                            "upos": POS_MAP[parts[3]],
                            **d
                        })

df = pd.DataFrame(ROWS)

# ======================================================
# FREQUENZE
# ======================================================
group_cols = [
    "corpus","token","lemma","upos",
    "affisso_principale","affissi_tutti",
    "base","id","wfp1","wfp2","wfp3","wfp4","wfp5","wfp6"
]

df_freq = df.groupby(group_cols).size().reset_index(name="frequenza")

# ======================================================
# CSV GLOBALI (COME PRIMA)
# ======================================================
df_freq.to_csv(os.path.join(OUTPUT,"FORME_LEMMI_DERIVATARIO.csv"), index=False)

df_forms = df_freq[["corpus","token","upos","affisso_principale","frequenza"]]
df_lemmas = df_freq[["corpus","lemma","upos","affisso_principale","frequenza"]]

df_forms.to_csv(os.path.join(OUTPUT,"TUTTE_LE_CARTELLE_FORME.csv"), index=False)
df_lemmas.to_csv(os.path.join(OUTPUT,"TUTTE_LE_CARTELLE_LEMMI.csv"), index=False)

# ======================================================
# COLLASSAMENTO
# ======================================================
def collapse(df, unit):
    return (
        df.groupby(["corpus",unit,"upos"])
        .agg({
            "frequenza":"sum",
            "affisso_principale": lambda x: "|".join(sorted(set(x)))
        })
        .reset_index()
    )

collapse(df_forms,"token").to_csv(
    os.path.join(OUTPUT,"FORME_COLLASSATE.csv"), index=False
)
collapse(df_lemmas,"lemma").to_csv(
    os.path.join(OUTPUT,"LEMMI_COLLASSATI.csv"), index=False
)

# ======================================================
# STATISTICHE (come prima, ma corrette)
# ======================================================
stats = []

for corpus,tot in TOTALI.items():
    sub = df_freq[df_freq["corpus"]==corpus]
    if sub.empty:
        continue
    for pos in POS_SET:
        sp = sub[sub["upos"]==pos]
        deriv = sp["frequenza"].sum()
        suff = sp[sp["affisso_principale"].str.endswith(("ARE","ERE","IRE","ICO","ALE"))]["frequenza"].sum()
        pref = deriv - suff
        stats.append({
            "corpus": corpus,
            "POS": pos,
            "derivati": deriv,
            "% derivati / lemmi": deriv / tot["lemmi"] * 100,
            "% derivati / token": deriv / tot["token"] * 100,
            "% suffissati / derivati": suff / deriv * 100 if deriv else 0,
            "% prefissati / derivati": pref / deriv * 100 if deriv else 0
        })

pd.DataFrame(stats).to_csv(
    os.path.join(OUTPUT,"STATISTICHE_PERCENTUALI.csv"),
    index=False
)

print("✔ COMPLETATO: output identico + derivatario integrato")
