import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}
OUTPUT_CSV = "risultati_mdd.csv"
MAX_LENGTH_THRESHOLD = 50 

def process_corpora():
    print(f"Calcolo Mean Dependency Distance (MDD)...")
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in IGNORE_FOLDERS: continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        if folder_name == "isdt":
                            if "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                                continue 
                        yield sentence, folder_name

def analyze_mdd():
    stats = {}

    for sentence, corpus in process_corpora():
        # Filtro lunghezza (per coerenza con le altre analisi)
        valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
        words_only = [t for t in valid_tokens if t['deprel'] != 'punct']
        sent_len = len(words_only)
        
        if sent_len == 0 or sent_len > MAX_LENGTH_THRESHOLD:
            continue
            
        # Calcolo Distanza per ogni dipendenza
        total_distance = 0
        n_dependencies = 0
        
        for token in words_only:
            head_id = token['head']
            token_id = token['id']
            
            # Saltiamo la radice (che ha head=0) perché non ha una distanza "fisica" da una parola
            if head_id == 0:
                continue
                
            # Calcolo distanza assoluta (|Head - Dependent|)
            dist = abs(head_id - token_id)
            
            total_distance += dist
            n_dependencies += 1
            
        if n_dependencies > 0:
            mdd = total_distance / n_dependencies
            
            if corpus not in stats:
                stats[corpus] = {'Sum_MDD': 0, 'Count': 0}
            
            stats[corpus]['Sum_MDD'] += mdd
            stats[corpus]['Count'] += 1

    # Export
    csv_rows = []
    for corpus, data in stats.items():
        if data['Count'] > 0:
            avg_mdd = data['Sum_MDD'] / data['Count']
        else:
            avg_mdd = 0
            
        csv_rows.append({
            'Corpus': corpus,
            'Mean Dependency Distance (MDD)': round(avg_mdd, 2),
            'N_Frasi': data['Count']
        })

    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by='Mean Dependency Distance (MDD)', ascending=False)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Analisi MDD completata. Risultati in {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df = analyze_mdd()
    print(df.to_string())