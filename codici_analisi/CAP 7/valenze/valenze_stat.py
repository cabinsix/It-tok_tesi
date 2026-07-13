import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
VALPAL_PATH = "database_valpal.csv"
MAX_LEN = 50

COL_LEMMA = "lemma"
COL_FRAME = "valency"

REALIZED_ARGS = {'nsubj', 'nsubj:pass', 'csubj', 'obj', 'iobj', 'ccomp', 'xcomp'}

# --- FUNZIONE DI SUPPORTO VALENZA (Usa la tua regex o logica attuale) ---
def conta_argomenti_complessa(frame_text):
    if not frame_text or pd.isna(frame_text): return 0
    import re
    numeri = re.findall(r'\d+', str(frame_text))
    if not numeri: return 0
    return max([int(n) for n in numeri])

def load_valpal_dict():
    try:
        df = pd.read_csv(VALPAL_PATH, sep=None, engine='python') 
    except Exception as e:
        print(f"Errore lettura ValPal: {e}")
        return {}
    valency_map = {}
    for _, row in df.iterrows():
        lemma = str(row[COL_LEMMA]).strip().lower()
        n_args = conta_argomenti_complessa(row[COL_FRAME])
        if lemma not in valency_map or n_args > valency_map[lemma]:
            valency_map[lemma] = n_args
    return valency_map

# ==========================================
# 1. PARSING E COSTRUZIONE DATASET COMPLETO
# ==========================================
def genera_dataset_token():
    val_db = load_valpal_dict()
    if not val_db: 
        print("Impossibile procedere senza database ValPal.")
        return pd.DataFrame()

    # Questa lista conterrà una riga per OGNI singolo token verbale trovato
    all_verb_tokens = []

    print("Estrazione token dai corpora...")
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in {"markit", "Ittok", "gen", "polso", "postwita","tr-it-tok","kiparla"}: continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        sent_id = sentence.metadata.get("sent_id", "unknown")
                        if folder_name == "isdt" and "isst_tanl" not in sent_id:
                            continue
                        
                        tokens = [t for t in sentence if isinstance(t['id'], int)]
                        if len(tokens) > MAX_LEN: continue
                        
                        # Mappa dipendenze figli
                        children_map = {}
                        for t in tokens:
                            if t['head'] != 0:
                                children_map.setdefault(t['head'], []).append(t)

                        # Analisi Verbi
                        for t in tokens:
                            if t['upos'] != 'VERB': continue
                            
                            lemma = t['lemma'].lower()
                            if lemma not in val_db: continue
                            
                            expected = val_db[lemma]
                            if expected == 0: continue
                            
                            observed = 0
                            if t['id'] in children_map:
                                for child in children_map[t['id']]:
                                    if child['deprel'] in REALIZED_ARGS:
                                        observed += 1
                            
                            if observed > expected: observed = expected
                            
                            sat_rate = (observed / expected) * 100
                            
                            # Salva i metadati del singolo token per gli effetti misti
                            all_verb_tokens.append({
                                "Corpus": folder_name,
                                "Sentence_ID": sent_id,
                                "Lemma": lemma,
                                "Sat_Rate": sat_rate
                            })

    df = pd.DataFrame(all_verb_tokens)
    print(f"Dataset creato con successo! Totale token estratti: {len(df)}")
    return df

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Genera il dataset non aggregato
    df_tokens = genera_dataset_token()
    
    if not df_tokens.empty:
        
        # Stampa le medie descrittive di controllo
        print("\n--- Medie Descrittive Rigenerate ---")
        print(df_tokens.groupby("Corpus")["Sat_Rate"].mean().round(2))
        
        # ==========================================
        # 2. VISUALIZZAZIONE DI CONTROLLO: BOXPLOT
        # ==========================================
        print("\nGenerazione Boxplot in corso...")
        plt.figure(figsize=(10, 6))
        
        # Ordiniamo i corpora per media crescente per rendere il grafico leggibile
        ordine_corpora = df_tokens.groupby("Corpus")["Sat_Rate"].mean().sort_values().index
        
        sns.boxplot(
            x='Corpus', 
            y='Sat_Rate', 
            data=df_tokens, 
            order=ordine_corpora,
            palette="Set2",
            width=0.5
        )
        
        plt.title("Distribuzione dei Tassi di Saturazione Valenziale per Corpus\n(Ispezione di Normalità e Varianza)", fontweight='bold', pad=15)
        plt.ylabel("Saturazione Valenziale (%)", fontsize=11)
        plt.xlabel("Corpus", fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        # ==========================================
        # 3. STATISTICA: MODELLI A EFFETTI MISTI (LMM)
        # ==========================================
        print("\n--- Esecuzione Modello Lineare a Effetti Misti ---")
        print("Variabile Dipendente (Y): Sat_Rate")
        print("Effetto Fisso (X): Corpus")
        print("Effetto Casuale (Random Effect): Lemma (Intercetta casuale per risolvere la non-indipendenza)")
        
        # Scegliamo un corpus di riferimento (es. l'ISDT scritto o KIParla) 
        # Modificando il Reference, cambiano i confronti di output. Di default prende il primo in ordine alfabetico.
        
        # Formula: Sat_Rate ~ Corpus
        # groups='Lemma': dice al modello che i dati dentro lo stesso Lemma sono clusterizzati (correlati)
        model = smf.mixedlm("Sat_Rate ~ Corpus", df_tokens, groups=df_tokens["Lemma"])
        model_result = model.fit()
        
        # Stampa il report scientifico completo
        print(model_result.summary())