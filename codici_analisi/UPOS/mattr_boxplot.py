import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from conllu import parse_incr
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ======================================================
# PARAMETRI GLOBALI
# ======================================================
ROOT_FOLDER = "conllu_files"
WINDOW_SIZE = 500
OUTPUT_CSV = "mattr_lemmi_sottocartelle.csv"
OUTPUT_WINDOWS_CSV = "mattr_lemmi_finestre500.csv"

# ======================================================
# FUNZIONE DI FILTRAGGIO ROBUSTA
# ======================================================
def deve_essere_esclusa(nome_cartella):
    nome_clean = nome_cartella.lower().strip()
    
    # 1. Esclusione CHIRURGICA di "kiparla" (mantenendo kiparla2.18 o simili)
    if nome_clean == "kiparla":
        return True
        
    # 2. Esclusione delle cartelle di controllo trittok
    target_inutili = ["trittok", "tr-it", "tr_it"]
    if any(target in nome_clean for target in target_inutili):
        return True
        
    return False

# ======================================================
# FUNZIONI UTILITARIE (CONLLU & MATTR)
# ======================================================
def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

def mattr(tokens, window_size=100):
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ratios = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        ratios.append(len(set(window)) / window_size)
    return sum(ratios) / len(ratios)

def mattr_windows(tokens, window_size=100):
    values = []
    if len(tokens) < window_size:
        return values
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        value = len(set(window)) / window_size
        values.append(value)
    return values

# ======================================================
# ANALISI SOTTOCARTELLA
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

                        if not lemma or upos in ("PROPN", "PUNCT"):
                            continue

                        lemma = lemma.lower()
                        token_count += 1
                        lemma_set.add(lemma)
                        pos_token_counter[upos] += 1
                        pos_lemma_sets[upos].add(lemma)
                        all_lemmas.append(lemma)
        except Exception as e:
            print(f"✘ Errore file {conllu_file}: {e}")

    mattr_score = mattr(all_lemmas, window_size=WINDOW_SIZE)
    window_values = mattr_windows(all_lemmas, window_size=WINDOW_SIZE)
    
    for i, v in enumerate(window_values, start=1):
        all_mattr_windows.append({"cartella": subfolder, "window_id": i, "mattr": v})

    print(f"✔ Elaborata: {subfolder} | Token: {token_count} | MATTR: {mattr_score:.4f}")
    return {
        "cartella": subfolder,
        "token_totali": token_count,
        "lemmi_unici": len(lemma_set),
        "mattr_lemmi": mattr_score,
    }

# ======================================================
# ESECUZIONE PRINCIPALE
# ======================================================
if __name__ == "__main__":
    walk = next(os.walk(ROOT_FOLDER), (None, [], []))
    subfolders = walk[1]

    if not subfolders:
        print(f"Nessuna sottocartella trovata in {ROOT_FOLDER}")
    else:
        results = []
        all_mattr_windows = []

        # 1. Loop di elaborazione con primo livello di filtro
        for subfolder in sorted(subfolders):
            if deve_essere_esclusa(subfolder):
                print(f"⏩ [SKIP] Esclusa dai conteggi: {subfolder}")
                continue

            sub_path = os.path.join(ROOT_FOLDER, subfolder)
            res = stats_subfolder(sub_path, subfolder, all_mattr_windows)
            results.append(res)

        # Creazione dei DataFrame
        df_riepilogo = pd.DataFrame(results)
        df_windows = pd.DataFrame(all_mattr_windows)

        # 2. SECONDO LIVELLO DI FILTRO (Antidoto contro i vecchi dati residui nel DataFrame)
        if not df_windows.empty:
            df_windows["_tmp_clean"] = df_windows["cartella"].astype(str).str.lower().str.strip()
            
            # Teniamo solo le righe che NON devono essere escluse
            condizione_conferma = df_windows["_tmp_clean"].apply(lambda x: not deve_essere_esclusa(x))
            df_windows = df_windows[condizione_conferma].drop(columns=["_tmp_clean"])

        # Salvataggio CSV puliti
        df_riepilogo.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        df_windows.to_csv(OUTPUT_WINDOWS_CSV, index=False, encoding="utf-8")
        print(f"\n✅ File CSV aggiornati e sovrascritti con successo.")

        # 3. Calcolo dell'ordine basato sulla MEDIANA del MATTR
        ordine_cartelle = (
            df_windows.groupby("cartella")["mattr"]
            .median()
            .sort_values(ascending=False)
            .index
        )

        print("📊 Corpora inclusi nel Boxplot definitivo:", list(ordine_cartelle))

        # 4. Generazione del Grafico
        plt.figure(figsize=(11, 7))
        sns.set_theme(style="whitegrid")

        sns.boxplot(
            data=df_windows,
            x="cartella",
            y="mattr",
            order=ordine_cartelle,
            palette="YlGnBu_r",
            width=0.45,
            fliersize=2.5,
        )

        plt.title(
            "Distribuzione del MATTR per Sottocartella\n(Finestra mobile = 500 token, ordinato per mediana decrescente)",
            fontsize=13,
            pad=15,
            fontweight="bold",
        )
        plt.xlabel("Sottocartella / Corpus", fontsize=11, labelpad=10)
        plt.ylabel("Moving-Average Type-Token Ratio (MATTR)", fontsize=11, labelpad=10)

        plt.xticks(rotation=25, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()