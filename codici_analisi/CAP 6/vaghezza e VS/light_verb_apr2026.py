import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"                # Cartella dei corpora
VSUPP_PATH = "pred_analitici_expanded.csv"       # Il CSV con la colonna 'vsupp'
OUTPUT_CSV = "verbi_supporto_count.csv"
         

COL_VSUPP = "vsupp"                      # Nome della colonna dei verbi supporto nel CSV

def load_vsupp_list():
    """
    Carica la lista dei verbi supporto dal CSV.
    """
    try:
        df = pd.read_csv(VSUPP_PATH, sep=";")
    except Exception as e:
        print(f"Errore lettura CSV vsupp: {e}")
        return set()
    
    vsupp_list = set(df[COL_VSUPP].dropna().str.strip().str.lower())
    print(f"Verbi supporto caricati: {len(vsupp_list)}")
    return vsupp_list

def analyze_support_verbs():
    vsupp_list = load_vsupp_list()
    if not vsupp_list:
        return
    
    stats = {}

    print("Analisi corpora in corso...")
    for root, dirs, files in os.walk(ROOT_DIR):
        corpus_name = os.path.basename(root)
        if corpus_name in {"markit", "postwita"}:
            continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                
                # --- MODIFIED SECTION START ---
                try:
                    # utf-8-sig helps with files that might have a BOM (Byte Order Mark)
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        for sentence in parse_incr(f):
                            # Filtro specifico ISDT
                            if corpus_name == "isdt":
                                sent_id = sentence.metadata.get("sent_id", "")
                                if "isst" not in sent_id:
                                    continue
                            
                            tokens = [t for t in sentence if isinstance(t['id'], int)]

                            
                            for t in tokens:
                                if t['upos'] != 'VERB':
                                    continue
                                lemma = t['lemma'].lower()
                                if lemma in vsupp_list:
                                    if corpus_name not in stats:
                                        stats[corpus_name] = {}
                                    stats[corpus_name][lemma] = stats[corpus_name].get(lemma, 0) + 1
                except UnicodeDecodeError:
                    # If UTF-8 fails, try Latin-1 (very common for older European datasets)
                    print(f"⚠️ UTF-8 fallito per {filename}, provo con Latin-1...")
                    with open(file_path, "r", encoding="latin-1") as f:
                        # ... (repeat the logic or wrap it in a helper function)
                        pass 
                except Exception as e:
                    print(f"❌ Errore durante la lettura di {filename}: {e}")
                # --- MODIFIED SECTION END ---

    # ... (rest of your CSV preparation code)
    # Preparazione CSV
    csv_rows = []
    for corpus, verbs in stats.items():
        for verb, count in verbs.items():
            csv_rows.append({
                'Corpus': corpus,
                'Verbo Supporto': verb,
                'Occorrenze': count
            })

    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by=['Corpus', 'Occorrenze'], ascending=[True, False])
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.to_string())

if __name__ == "__main__":
    analyze_support_verbs()
