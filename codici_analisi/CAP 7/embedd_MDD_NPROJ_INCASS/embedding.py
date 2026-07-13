import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}
OUTPUT_CSV = "risultati_clause_embedding.csv"
MAX_LENGTH_THRESHOLD = 50  # Filtro coerenza

# Relazioni che fanno scattare il livello successivo (+1)
SUBORDINATE_RELS = {
    'acl', 'acl:relcl', 
    'advcl', 
    'ccomp', 
    'xcomp', 
    'csubj', 'csubj:pass'
}

def process_corpora():
    print(f"Calcolo Profondità Clausale (Subordinate Embedding)...")
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

def calculate_clause_depth(sentence):
    """
    Calcola la profondità di incassamento delle CLAUSOLE.
    Root = 0.
    Subordinata = Padre + 1.
    Coordinata = Padre + 0.
    Altro = Padre + 0.
    """
    children_map = {}
    roots = []
    valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
    
    for token in valid_tokens:
        head_id = token['head']
        if head_id == 0:
            roots.append(token['id'])
        else:
            if head_id not in children_map: children_map[head_id] = []
            children_map[head_id].append(token)

    # Stack: (token_id, current_clause_depth)
    # Partiamo dalla radice a livello 0
    stack = [(r, 0) for r in roots]
    
    max_depth_reached = 0
    
    while stack:
        curr_id, curr_depth = stack.pop()
        
        # Aggiorniamo il record mondiale di profondità per questa frase
        if curr_depth > max_depth_reached:
            max_depth_reached = curr_depth
            
        if curr_id in children_map:
            for child in children_map[curr_id]:
                child_id = child['id']
                deprel = child['deprel']
                
                # --- LOGICA DI CALCOLO ---
                if deprel in SUBORDINATE_RELS:
                    # È una subordinata: scendiamo di un livello
                    next_depth = curr_depth + 1
                elif deprel == 'conj':
                    # È una coordinata: restiamo allo stesso livello (Skip)
                    next_depth = curr_depth
                else:
                    # È un altro elemento (sogg, ogg...): restiamo allo stesso livello
                    next_depth = curr_depth
                
                stack.append((child_id, next_depth))
                
    return max_depth_reached

def analyze_clause_embedding():
    stats = {}

    for sentence, corpus in process_corpora():
        # Filtro lunghezza (coerenza)
        valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
        words_only = [t for t in valid_tokens if t['deprel'] != 'punct']
        sent_len = len(words_only)
        
        if sent_len == 0 or sent_len > MAX_LENGTH_THRESHOLD:
            continue
            
        # Calcolo profondità massima clausale della frase
        max_clause_depth = calculate_clause_depth(sentence)
        
        if corpus not in stats:
            stats[corpus] = {'Sum_Max_Clause_Depth': 0, 'Count_Frasi': 0}
            
        stats[corpus]['Sum_Max_Clause_Depth'] += max_clause_depth
        stats[corpus]['Count_Frasi'] += 1

    # Export
    csv_rows = []
    for corpus, data in stats.items():
        if data['Count_Frasi'] > 0:
            avg_embedding = data['Sum_Max_Clause_Depth'] / data['Count_Frasi']
        else:
            avg_embedding = 0
            
        csv_rows.append({
            'Corpus': corpus,
            'Avg Max Clause Embedding': round(avg_embedding, 2),
            'N_Frasi': data['Count_Frasi']
        })

    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by='Avg Max Clause Embedding', ascending=False)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Analisi Embedding Clausale completata. Risultati in {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df = analyze_clause_embedding()
    print(df.to_string())