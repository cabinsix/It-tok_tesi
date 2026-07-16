import os
import pandas as pd
import csv
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
VALPAL_PATH = "database_valpal.csv" # Il tuo file ValPal
OUTPUT_CSV = "risultati_saturazione_valpal.csv"
MAX_LEN = 50

# Configura qui i nomi delle colonne del TUO file ValPal
COL_LEMMA = "lemma"       # Come si chiama la colonna del verbo?
COL_FRAME = "valency" # Come si chiama la colonna della struttura?

# Mappa per contare gli argomenti dalle etichette testuali di ValPal
# Esempio: se ValPal scrive "nom.acc.dat", contiamo i punti + 1, o usiamo logiche specifiche.
# Se hai già il numero, salta questa parte.
import re

def conta_argomenti_complessa(frame_text):
    """
    Decifra notazioni complesse tipo: '1 > V.subj[1] > 2 (a+3)'
    Restituisce il numero massimo trovato che indica un argomento.
    """
    if not frame_text or pd.isna(frame_text): return 0
    text = str(frame_text)
    
    # 1. Strategia Euristica: Cerchiamo i numeri "nudi" che indicano argomenti (1, 2, 3, 4)
    # Spesso in queste notazioni i numeri degli argomenti sono 1, 2, 3...
    # Le parentesi quadre [1] spesso indicano il legame semantico, le ignoriamo per sicurezza o le trattiamo uguale.
    
    # Cerchiamo tutti i numeri interi nella stringa
    numeri = re.findall(r'\d+', text)
    
    if not numeri:
        return 0
    
    # Convertiamo in interi
    numeri_int = [int(n) for n in numeri]
    
    # La valenza è solitamente il numero più alto menzionato
    # Es: in "1 > V > 2 (a+3)" il max è 3.
    valenza_max = max(numeri_int)
    
    return valenza_max

# --- TESTIAMOLA SUBITO ---
test_strings = [
    "1 > V.subj[1]",              # Intransitivo (Valenza 1)
    "1 > V.subj[1] > 2",          # Transitivo (Valenza 2)
    "1 > V.subj[1] > 2 (a+3)",    # Ditransitivo (Valenza 3)
    "0 > V.impers",               # Impersonale (Valenza 0)
    "subj-obj",                   # Notazione testuale (Fallisce con regex, serve logica 'len')
]

for s in test_strings:
    print(f"'{s}' -> Valenza: {conta_argomenti_complessa(s)}")

def load_valpal_dict():
    print("Caricamento ValPal...")
    try:
        # Legge il CSV (o Excel se cambi in read_excel)
        df = pd.read_csv(VALPAL_PATH, sep=None, engine='python') 
    except Exception as e:
        print(f"Errore lettura ValPal: {e}")
        return {}

    valency_map = {}
    
    for _, row in df.iterrows():
        lemma = str(row[COL_LEMMA]).strip().lower()
        frame = row[COL_FRAME]
        
        n_args = conta_argomenti_complessa(frame)
        
        # Gestione duplicati: teniamo la valenza MASSIMA per quel verbo
        if lemma not in valency_map:
            valency_map[lemma] = n_args
        else:
            if n_args > valency_map[lemma]:
                valency_map[lemma] = n_args
                
    print(f"Database caricato: {len(valency_map)} verbi unici.")
    return valency_map

# Relazioni UD che contiamo come ARGOMENTI REALIZZATI
REALIZED_ARGS = {'nsubj', 'nsubj:pass', 'csubj', 'obj', 'iobj', 'ccomp', 'xcomp'}

def analyze_saturation_valpal():
    # 1. Carica il dizionario
    val_db = load_valpal_dict()
    if not val_db: return

    stats = {}

    print("Analisi corpora in corso...")
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in {"markit", "Ittok", "gen", "polso", "postwita"}: continue
        
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        # Filtri soliti
                        if folder_name == "isdt" and "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                            continue
                        tokens = [t for t in sentence if isinstance(t['id'], int)]
                        if len(tokens) > MAX_LEN: continue
                        
                        # Mappa figli
                        children_map = {}
                        for t in tokens:
                            if t['head'] != 0:
                                children_map.setdefault(t['head'], []).append(t)

                        # Analisi Verbi
                        for t in tokens:
                            if t['upos'] != 'VERB': continue
                            
                            lemma = t['lemma'].lower()
                            
                            # Se il verbo non è nel ValPal, lo ignoriamo
                            if lemma not in val_db: continue
                            
                            expected = val_db[lemma]
                            # Ignoriamo verbi a valenza 0 (es. piovere) per evitare divisioni strane
                            if expected == 0: continue
                            
                            # Contiamo gli osservati
                            observed = 0
                            if t['id'] in children_map:
                                for child in children_map[t['id']]:
                                    if child['deprel'] in REALIZED_ARGS:
                                        observed += 1
                            
                            # Capping: se ne trovi più del previsto (aggiunti), fermati a expected
                            # (Opzionale: puoi toglierlo se vuoi vedere l'iper-saturazione)
                            if observed > expected: observed = expected
                            
                            if folder_name not in stats:
                                stats[folder_name] = {'Sum_Sat_Rate': 0, 'Count': 0}
                            
                            # Calcolo saturazione singola istanza
                            sat_rate = (observed / expected) * 100
                            
                            stats[folder_name]['Sum_Sat_Rate'] += sat_rate
                            stats[folder_name]['Count'] += 1

    # Export
    csv_rows = []
    for corpus, data in stats.items():
        if data['Count'] > 0:
            avg_sat = data['Sum_Sat_Rate'] / data['Count']
        else:
            avg_sat = 0
            
        csv_rows.append({
            'Corpus': corpus,
            'Saturazione Media (%)': round(avg_sat, 2),
            'Verbi Analizzati': data['Count']
        })

    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by='Saturazione Media (%)', ascending=False)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.to_string())

if __name__ == "__main__":
    analyze_saturation_valpal()