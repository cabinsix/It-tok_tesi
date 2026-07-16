import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}
OUTPUT_CSV = "risultati_havelka_binned_50.csv"

# NUOVO LIMITE: Tagliamo tutto ciò che supera i 50 token.
# Questo garantisce che stiamo analizzando solo frasi reali e non errori.
MAX_LENGTH_THRESHOLD = 50 

def process_corpora():
    print(f"Calcolo Havelka (Binning max 50) NO PUNCT...")
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

def get_subtree(node_id, children_map):
    """Versione Iterativa (Anti-Crash)"""
    subtree = {node_id}
    stack = [node_id]
    visited = {node_id}
    while stack:
        curr = stack.pop()
        if curr in children_map:
            for child in children_map[curr]:
                c_id = child['id']
                if c_id not in visited:
                    visited.add(c_id)
                    subtree.add(c_id)
                    stack.append(c_id)
    return subtree

def analyze_havelka_strict_bins():
    stats = {}

    for sentence, corpus in process_corpora():
        # Otteniamo tutti i token reali
        all_tokens = [t for t in sentence if isinstance(t['id'], int)]
        
        # Calcoliamo lunghezza basandoci SOLO sulle parole (No Punct)
        words_only = [t for t in all_tokens if t['deprel'] != 'punct']
        sent_len = len(words_only)

        # --- FILTRO RIGIDO ---
        # Scartiamo frasi vuote o mostruose (> 50 parole)
        if sent_len == 0 or sent_len > MAX_LENGTH_THRESHOLD:
            continue

        # --- NUOVO BINNING ---
        if sent_len <= 15:
            bin_name = '1. Short (1-15)'
        elif sent_len <= 35:
            bin_name = '2. Medium (16-35)'
        else:
            # Qui finiscono solo le frasi tra 36 e 50 (grazie al filtro sopra)
            bin_name = '3. Long (36-50)'

        if corpus not in stats: stats[corpus] = {}
        if bin_name not in stats[corpus]:
            stats[corpus][bin_name] = {'Tot_Frasi': 0, 'Frasi_NonProj': 0, 
                                       'Tot_Archi': 0, 'Archi_NonProj': 0}

        # Analisi Havelka
        children_map = {}
        id_to_token = {}
        
        for token in all_tokens:
            t_id = token['id']
            head_id = token['head']
            id_to_token[t_id] = token
            
            if head_id != 0:
                if head_id not in children_map: children_map[head_id] = []
                children_map[head_id].append(token)

        is_sentence_nonproj = False
        nonproj_arcs_count = 0
        
        # Iteriamo sui token parola (j)
        for token_j in words_only:
            j = token_j['id']
            i = token_j['head']
            
            if i == 0: continue
            
            subtree_i = get_subtree(i, children_map)
            start, end = sorted((i, j))
            
            is_arc_nonproj = False
            
            # Controllo intervallo
            for v_id in range(start + 1, end):
                if v_id in id_to_token:
                    token_v = id_to_token[v_id]
                    
                    # Punteggiatura è trasparente
                    if token_v['deprel'] == 'punct':
                        continue
                        
                    # Violazione Havelka
                    if v_id not in subtree_i:
                        is_arc_nonproj = True
                        break
            
            if is_arc_nonproj:
                nonproj_arcs_count += 1
                is_sentence_nonproj = True

        # Aggiornamento
        stats[corpus][bin_name]['Tot_Frasi'] += 1
        stats[corpus][bin_name]['Tot_Archi'] += len(words_only)
        stats[corpus][bin_name]['Archi_NonProj'] += nonproj_arcs_count
        
        if is_sentence_nonproj:
            stats[corpus][bin_name]['Frasi_NonProj'] += 1

    # Export
    csv_rows = []
    for corpus, bins in stats.items():
        for bin_name, data in bins.items():
            if data['Tot_Frasi'] > 0:
                perc_frasi = round((data['Frasi_NonProj'] / data['Tot_Frasi']) * 100, 2)
            else: perc_frasi = 0.0
                
            if data['Tot_Archi'] > 0:
                perc_archi = round((data['Archi_NonProj'] / data['Tot_Archi']) * 100, 2)
            else: perc_archi = 0.0
            
            csv_rows.append({
                'Corpus': corpus,
                'Lunghezza': bin_name,
                '% Non-Proj (Frasi)': perc_frasi,
                '% Non-Proj (Archi)': perc_archi,
                'N_Frasi': data['Tot_Frasi']
            })

    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by=['Corpus', 'Lunghezza'])
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Analisi completata (Max 50 token). File: {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df = analyze_havelka_strict_bins()
    print(df.to_string())