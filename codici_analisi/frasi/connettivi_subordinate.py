import os
import pandas as pd
from conllu import parse_incr
from collections import Counter

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}

# Relazioni subordinate da analizzare
SUBORDINATE_DEPRELS = {'advcl', 'ccomp', 'xcomp', 'acl', 'acl:relcl', 'csubj'}

# Lemmi che sono sicuramente relativi se non sono mark
KNOWN_RELATIVES = {'cui', 'quale', 'quali', 'chi', 'quanto', 'quanti', 'dove', 'onde'}

OUTPUT_CSV = "risultati_connettivi_gold.csv"

def process_corpora():
    print(f"Elaborazione Gold Standard...")
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in IGNORE_FOLDERS: continue
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        if folder_name == "isdt":
                            if "isst_tanl" not in sentence.metadata.get("sent_id", ""): continue 
                        yield sentence, folder_name

def analyze_connectives_gold():
    corpus_connectives = {}
    
    for sentence, corpus in process_corpora():
        if corpus not in corpus_connectives: corpus_connectives[corpus] = Counter()

        children_map = {}
        for token in sentence:
            head = token['head']
            if head not in children_map: children_map[head] = []
            children_map[head].append(token)

        for token in sentence:
            clause_type = token['deprel'] # Es. acl:relcl, ccomp, advcl...
            
            if clause_type not in SUBORDINATE_DEPRELS:
                continue
            
            verb_id = token['id']
            found_connective = None
            
            if verb_id in children_map:
                children = children_map[verb_id]
                
                # Cerca candidati connettivi tra i figli
                mark_token = None
                rel_token = None
                
                for child in children:
                    lemma = child['lemma'].lower()
                    deprel = child['deprel']
                    
                    if deprel == 'mark':
                        mark_token = lemma
                    
                    # Se non è mark, potrebbe essere un relativo (soggetto/oggetto)
                    # Lo consideriamo candidato se è "che" o uno dei relativi noti
                    elif lemma == 'che' or lemma in KNOWN_RELATIVES:
                        # Escludiamo casi palesi di det/amod se non siamo in una relativa
                        if deprel not in ['det', 'amod']: 
                            rel_token = lemma

                # --- LOGICA DI DECISIONE ---
                
                # 1. Se c'è un MARK, vince quasi sempre lui (è una congiunzione)
                if mark_token:
                    if mark_token == 'che':
                        found_connective = "che (CONJ)"
                    else:
                        found_connective = mark_token
                
                # 2. Se non c'è MARK, controlliamo se è una RELATIVA STRUTTURALE (acl:relcl)
                elif clause_type == 'acl:relcl':
                    # In una relativa, se trovo "che" è un pronome relativo
                    if rel_token == 'che':
                        found_connective = "che (PRON)"
                    elif rel_token:
                        found_connective = rel_token # cui, quale...
                
                # 3. Caso Raro: "che" interrogativo indiretto (es. "So che fai")
                # Se siamo in ccomp/xcomp e troviamo un "che" non mark -> Probabile Pronome/Det interrogativo
                elif clause_type in ['ccomp', 'xcomp'] and rel_token == 'che':
                     found_connective = "che (PRON/DET)" # O lo uniamo a PRON

            final_label = found_connective if found_connective else "(NULL/Implicito)"
            corpus_connectives[corpus].update([final_label])

    # Export
    csv_rows = []
    for corpus, counter in corpus_connectives.items():
        total_subs = sum(counter.values())
        for connettivo, count in counter.most_common():
            csv_rows.append({
                'Corpus': corpus,
                'Connettivo': connettivo,
                'Frequenza': count,
                'Percentuale': round((count/total_subs)*100, 2)
            })

    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Analisi Gold salvata in {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze_connectives_gold()

import matplotlib.pyplot as plt
import seaborn as sns

    # ==========================
    # DataFrame finale
    # ==========================
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Analisi Gold salvata in {OUTPUT_CSV}")

    # ==========================================
    # HEATMAP 1: percentuale dei connettivi
    # ==========================================

    pivot = df.pivot(
        index="Corpus",
        columns="Connettivo",
        values="Percentuale"
    ).fillna(0)

    plt.figure(figsize=(14,6))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="YlGnBu",
        fmt=".1f",
        linewidths=.5
    )
    plt.title("Percentuale dei connettivi nei corpora")
    plt.tight_layout()
    plt.savefig("heatmap_connettivi.png", dpi=300)
    plt.show()

    # ==========================================
    # HEATMAP 2: confronto corpus vs corpus
    # ==========================================

    similarity = pivot.T.corr(method="pearson")

    plt.figure(figsize=(8,7))
    sns.heatmap(
        similarity,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=.5
    )
    plt.title("Similarità tra corpora (Pearson)")
    plt.tight_layout()
    plt.savefig("heatmap_corpora.png", dpi=300)
    plt.show()