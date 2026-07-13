import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}

# Mappatura Ruoli Sintattici
SOGG_RELS = {'nsubj', 'nsubj:pass'}
OBJ_RELS = {'obj'}
# Tutto il resto (obl, iobj, nmod, etc.) sarà 'NOBJ'

def get_np_weight(head_token, children):
    """Determina la categoria di pesantezza di un NP (Voghera 2005)"""
    
    # Categoria vi: Pronomi
    if head_token['upos'] == 'PRON':
        return "vi. PRON"
    
    # Se non è un nome o nome proprio, non lo consideriamo un NP standard per questa analisi
    if head_token['upos'] not in {'NOUN', 'PROPN'}:
        return None

    # Analisi dei figli
    has_det = any(c['deprel'] == 'det' for c in children)
    has_relcl = any(c['deprel'] == 'acl:relcl' for c in children)
    # Altri modificatori: aggettivi, modificatori nominali, numerali, apposizioni
    has_mod = any(c['deprel'] in {'amod', 'nmod', 'nummod', 'flat', 'compound', 'appos'} for c in children)

    # Gerarchia Voghera
    if has_det and has_relcl:
        return "i. DET+NP+REL"
    elif has_det and has_mod:
        return "ii. DET+NP+MOD"
    elif has_mod and not has_det:
        return "iii. NP+MOD"
    elif has_det and not has_mod and not has_relcl:
        return "iv. DET+NP"
    else:
        return "v. NP Nudo"

def analyze_np_heaviness():
    results = []

    for root, dirs, files in os.walk(ROOT_DIR):
        folder = os.path.basename(root)
        if folder in IGNORE_FOLDERS: continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        # Filtro ISDT
                        if folder == "isdt" and "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                            continue
                        
                        tokens = [t for t in sentence if isinstance(t['id'], int)]
                        
                        # Mappa figli
                        children_map = {}
                        for t in tokens:
                            if t['head'] != 0:
                                children_map.setdefault(t['head'], []).append(t)
                        
                        for t in tokens:
                            # Identifichiamo i potenziali capi di un NP
                            if t['upos'] in {'NOUN', 'PROPN', 'PRON'}:
                                deprel = t['deprel'].split(':')[0] # Pulizia sub-labels
                                
                                # Determiniamo il ruolo
                                if t['deprel'] in SOGG_RELS:
                                    role = "SOGG"
                                elif t['deprel'] in OBJ_RELS:
                                    role = "OBJ"
                                else:
                                    role = "NOBJ"
                                
                                # Calcoliamo la pesantezza
                                children = children_map.get(t['id'], [])
                                weight = get_np_weight(t, children)
                                
                                if weight:
                                    results.append({
                                        'Corpus': folder,
                                        'Ruolo': role,
                                        'Pesantezza': weight
                                    })

    df = pd.DataFrame(results)
    
    # Creiamo una tabella pivot con le percentuali
    summary = pd.crosstab([df.Corpus, df.Ruolo], df.Pesantezza, normalize='index') * 100
    return summary.round(2)

# --- ESECUZIONE ---
df_heavy = analyze_np_heaviness()
print(df_heavy.to_string())