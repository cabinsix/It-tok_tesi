import os
from collections import Counter, defaultdict
from conllu import parse_incr
from tqdm import tqdm
import pandas as pd

# ======================================================
# PARAMETRI
# ======================================================
ROOT_FOLDER = "conllu_files"
WINDOW_SIZE = 500
OUTPUT_CSV = "mattr_lemmi_sottocartelle.csv"
OUTPUT_WINDOWS_CSV = "mattr_lemmi_finestre500.csv"  # ✅ NUOVO

# ======================================================
# FUNZIONE PER ITERARE I FILE CONLLU
# ======================================================
def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

# ======================================================
# FUNZIONE MATTR MEDIA (INVARIATA)
# ======================================================
def mattr(tokens, window_size=100):
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ratios = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        ratios.append(len(set(window)) / window_size)
    return sum(ratios) / len(ratios)

# ======================================================
# FUNZIONE MATTR PER OGNI FINESTRA (NUOVA)
# ======================================================
def mattr_windows(tokens, window_size=100):
    values = []
    if len(tokens) < window_size:
        return values
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        value = len(set(window)) / window_size
        values.append(value)
    return values

# ======================================================
# FUNZIONE PER STATISTICHE PER SOTTOCARTELLA
# ======================================================
def stats_subfolder(sub_path, subfolder, all_mattr_windows):
    lemma_set = set()
    pos_token_counter = Counter()
    pos_lemma_sets = defaultdict(set)
    all_lemmas = []

    token_count = 0
    is_isdt = subfolder.lower() == "isdt"

    for conllu_file in tqdm(iter_conllu_files(sub_path), desc=f"Lettura {subfolder}", unit="file"):
        try:
            with open(conllu_file, "r", encoding="utf-8", errors="replace") as f:
                for sentence in parse_incr(f):

                    if is_isdt:
                        sent_id = sentence.metadata.get("sent_id", "")
                        if "isst_tanl" not in sent_id:
                            continue

                    for token in sentence:
                        if not isinstance(token.get("id"), int):
                            continue

                        lemma = token.get("lemma")
                        upos = token.get("upos") or "_"

                        if not lemma:
                            continue

                        if upos in ("PROPN", "PUNCT"):
                            continue

                        lemma = lemma.lower()
                        token_count += 1
                        lemma_set.add(lemma)
                        pos_token_counter[upos] += 1
                        pos_lemma_sets[upos].add(lemma)
                        all_lemmas.append(lemma)

        except Exception as e:
            print(f"✘ Errore file {conllu_file}: {e}")

    # =========================
    # MATTR
    # =========================
    mattr_score = mattr(all_lemmas, window_size=WINDOW_SIZE)

    # ✅ MATTR PER OGNI FINESTRA
    window_values = mattr_windows(all_lemmas, window_size=WINDOW_SIZE)
    for i, v in enumerate(window_values, start=1):
        all_mattr_windows.append({
            "cartella": subfolder,
            "window_id": i,
            "mattr": v
        })

    # =========================
    # OUTPUT A SCHERMO (INVARIATO)
    # =========================
    print(f"\n==============================")
    print(f"SOTTOCARTELLA: {subfolder}")
    if is_isdt:
        print("⚠️  ISDT: solo frasi con sent_id contenente 'isst_tanl'")
    print(f"Token totali (esclusi PROPN e PUNCT): {token_count}")
    print(f"Lemmi unici totali (esclusi PROPN e PUNCT): {len(lemma_set)}")
    print(f"MATTR lemmi (window={WINDOW_SIZE}, esclusi PROPN e PUNCT): {mattr_score:.4f}")
    print("\nPOS | Token | Lemmi unici")
    print("-" * 32)
    for upos, token_freq in pos_token_counter.most_common():
        print(f"{upos:6s} | {token_freq:6d} | {len(pos_lemma_sets[upos]):12d}")

    return {
        "cartella": subfolder,
        "token_totali": token_count,
        "lemmi_unici": len(lemma_set),
        "mattr_lemmi": mattr_score
    }

# ======================================================
# MAIN
# ======================================================
walk = next(os.walk(ROOT_FOLDER), (None, [], []))
subfolders = walk[1]

if not subfolders:
    print(f"Nessuna sottocartella trovata in {ROOT_FOLDER}")
else:
    results = []
    all_mattr_windows = []  # ✅ CONTENITORE GLOBALE

    for subfolder in sorted(subfolders):
        sub_path = os.path.join(ROOT_FOLDER, subfolder)
        res = stats_subfolder(sub_path, subfolder, all_mattr_windows)
        results.append(res)

    # CSV riepilogativo (medie)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ CSV riepilogativo creato: {OUTPUT_CSV}")

    # CSV con TUTTE le finestre MATTR
    df_windows = pd.DataFrame(all_mattr_windows)
    df_windows.to_csv(OUTPUT_WINDOWS_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV finestre MATTR creato: {OUTPUT_WINDOWS_CSV}")
