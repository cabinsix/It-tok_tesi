import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}
SOGG_RELS = {'nsubj', 'nsubj:pass'}
OBJ_RELS = {'obj'}

def get_subtree_tokens(root_token, children_map):
    """Calcola la dimensione del sintagma in modo iterativo (evita RecursionError)"""
    subtree_ids = set()
    stack = [root_token['id']]
    visited = set() # Per evitare cicli infiniti in caso di errori nel CoNLL-U

    while stack:
        current_id = stack.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        subtree_ids.add(current_id)
        
        # Aggiungiamo i figli allo stack
        if current_id in children_map:
            for child in children_map[current_id]:
                # Escludiamo 'conj' per non mischiate NP diversi (come concordato)
                # Escludiamo anche 'punct' per una misura più pulita del peso
                if child['deprel'] not in {'conj', 'punct'}:
                    stack.append(child['id'])
                    
    return list(subtree_ids)

def analyze_advanced_heaviness():
    results = []

    for root, dirs, files in os.walk(ROOT_DIR):
        folder = os.path.basename(root)
        if folder in IGNORE_FOLDERS: continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        if folder == "isdt" and "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                            continue
                        
                        tokens = [t for t in sentence if isinstance(t['id'], int)]
                        children_map = {}
                        for t in tokens:
                            if t['head'] != 0:
                                children_map.setdefault(t['head'], []).append(t)
                        
                        for t in tokens:
                            if t['upos'] in {'NOUN', 'PROPN', 'PRON'}:
                                # 1. Determinazione Ruolo
                                if t['deprel'] in SOGG_RELS: role = "SOGG"
                                elif t['deprel'] in OBJ_RELS: role = "OBJ"
                                else: role = "NOBJ"
                                
                                # 2. Analisi Strutturale (Voghera)
                                children = children_map.get(t['id'], [])
                                has_det = any(c['deprel'] == 'det' for c in children)
                                has_relcl = any(c['deprel'] == 'acl:relcl' for c in children)
                                mods = [c for c in children if c['deprel'] in {'amod', 'nmod', 'nummod', 'flat', 'appos'}]
                                
                                # Categoria Voghera
                                if t['upos'] == 'PRON': weight_cat = "vi. PRON"
                                elif has_det and has_relcl: weight_cat = "i. DET+NP+REL"
                                elif has_det and mods: weight_cat = "ii. DET+NP+MOD"
                                elif mods and not has_det: weight_cat = "iii. NP+MOD"
                                elif has_det: weight_cat = "iv. DET+NP"
                                else: weight_cat = "v. NP Nudo"

                                # 3. Analisi Avanzata (Linearità e Ricorsione)
                                subtree_ids = get_subtree_tokens(t, children_map)
                                token_length = len(set(subtree_ids)) # Lunghezza in token
                                mod_count = len(mods) + (1 if has_relcl else 0) # Numero modificatori diretti
                                
                                results.append({
                                    'Corpus': folder,
                                    'Ruolo': role,
                                    'Categoria': weight_cat,
                                    'Lunghezza_Token': token_length,
                                    'Num_Modificatori': mod_count
                                })

    df = pd.DataFrame(results)
    
    # Aggregazione finale: Media lunghezza per ogni categoria e corpus
    summary = df.groupby(['Corpus', 'Ruolo', 'Categoria']).agg({
        'Categoria': 'count',
        'Lunghezza_Token': 'mean',
        'Num_Modificatori': 'mean'
    }).rename(columns={'Categoria': 'Frequenza'}).reset_index()
    
    return summary

# Esecuzione
df_adv = analyze_advanced_heaviness()
print(df_adv.to_string())