import os
import pandas as pd
from conllu import parse_incr
import scipy.stats as stats
import statsmodels.formula.api as smf
import numpy as np

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita","kiparla","tr-it-tok"}
MAX_LENGTH_THRESHOLD = 50  # Filtro coerenza

SUBORDINATE_RELS = {
    'acl', 'acl:relcl', 
    'advcl', 
    'ccomp', 
    'xcomp', 
    'csubj', 'csubj:pass'
}

def process_corpora():
    print("Lettura dei corpora in corso...")
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in IGNORE_FOLDERS: 
            continue
        
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
    """Calcola la profondità massima di incassamento clausale."""
    children_map = {}
    roots = []
    valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
    
    for token in valid_tokens:
        head_id = token['head']
        if head_id == 0:
            roots.append(token['id'])
        else:
            if head_id not in children_map: 
                children_map[head_id] = []
            children_map[head_id].append(token)

    stack = [(r, 0) for r in roots]
    max_depth_reached = 0
    visited = set()
    
    while stack:
        curr_id, curr_depth = stack.pop()
        
        if curr_id in visited:
            continue
        visited.add(curr_id)
        
        if curr_depth > max_depth_reached:
            max_depth_reached = curr_depth
            
        if curr_id in children_map:
            for child in children_map[curr_id]:
                child_id = child['id']
                deprel = child['deprel']
                
                if deprel in SUBORDINATE_RELS:
                    next_depth = curr_depth + 1
                else:
                    next_depth = curr_depth
                
                stack.append((child_id, next_depth))
                
    return max_depth_reached

def build_sentence_dataframe():
    """Crea il dataset frase per frase con Lunghezza e Profondità."""
    dataset = []

    for sentence, corpus in process_corpora():
        valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
        words_only = [t for t in valid_tokens if t['deprel'] != 'punct']
        sent_len = len(words_only)
        
        if sent_len == 0 or sent_len > MAX_LENGTH_THRESHOLD:
            continue
            
        max_clause_depth = calculate_clause_depth(sentence)
        
        dataset.append({
            'Corpus': corpus,
            'Length': sent_len,
            'Depth': max_clause_depth
        })

    return pd.DataFrame(dataset)

def run_advanced_stats(df):
    print("\n" + "="*50)
    print("   ANALISI AVANZATA: CORRELAZIONE E REGRESSIONE")
    print("="*50)
    
    # 1. CORRELAZIONE DI SPEARMAN (Lunghezza vs Profondità)
    rho, p_val_spearman = stats.spearmanr(df['Length'], df['Depth'])
    print(f"\n[1] CORRELAZIONE DI SPEARMAN")
    print(f"Coefficiente rho di Spearman: {rho:.4f}")
    print(f"p-value:                      {p_val_spearman:.4e}")
    print("Interpretazione: Un rho positivo indica che all'aumentare della lunghezza aumenta la profondità.")

    # 2. REGRESSIONE LINEARE (Controllando per la Lunghezza)
    print("\n[2] MODELLO DI REGRESSIONE LINEARE (ANCOVA)")
    # La formula esprime: La Profondità dipende dalla Lunghezza + dal tipo di Corpus
    # C(Corpus) dice a Python di trattare la variabile come categorie discrete
    model = smf.ols("Depth ~ Length + C(Corpus)", data=df).fit()
    
    # Stampiamo i coefficienti principali per facilitare la lettura
    print("\nRisultati sintetici del Modello:")
    summary_df = pd.DataFrame({
        'Coefficiente (Impatto)': model.params,
        'p-value': model.pvalues
    })
    print(summary_df.to_string())
    
    print("\n" + "-"*50)
    print("NOTA PER LA LETTURA :")
    print("- Il coefficiente di 'Length' indica di quanto aumenta la profondità per ogni parola in più.")
    print("- I coefficienti 'C(Corpus)[T....]' mostrano la differenza di quel corpus rispetto a quello di riferimento.")
    print("- Se il p-value del corpus è < 0.05, quel corpus è significativamente più complesso (o meno) AL NETTO della lunghezza.")
    print("-"*50)

if __name__ == "__main__":
    # Estrae i dati e crea il DataFrame
    df_frasi = build_sentence_dataframe()
    
    if not df_frasi.empty:
        # Esegue i test di Spearman e la Regressione di controllo
        run_advanced_stats(df_frasi)
        
        # Mostra comunque le medie descrittive alla fine
        print("\n[3] MEDIE DESCRITTIVE DI RIFERIMENTO:")
        print(df_frasi.groupby('Corpus')[['Depth', 'Length']].mean().round(2))
    else:
        print("Nessun dato trovato. Controlla la cartella dei file CoNLL-U.")