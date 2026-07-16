import os
import re
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def count_clean_tokens(text):
    """
    Calcola il numero di token puliti (senza link, punteggiatura o hashtag/menzioni isolati)
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    # Rimuove link (http/https/www)
    text_no_links = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Rimuove punteggiatura e underscore
    text_clean = re.sub(r'[^\w\s]|_', ' ', text_no_links)
    # Ritorna il conteggio delle parole rimaste
    words = text_clean.strip().split()
    return len(words)

def parse_conllu(filepath):
    """
    Legge un file CoNLL-U preservando i metadati e calcolando i token puliti.
    """
    sentences = []
    current_sent = {
        'sent_id': None,
        'text': "",
        'n_token': 0,
        'raw_lines': []
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            
            if not line_stripped:  # Riga vuota = fine frase
                if current_sent['sent_id'] is not None:
                    current_sent['clean_tokens'] = count_clean_tokens(current_sent['text'])
                    sentences.append(current_sent)
                current_sent = {
                    'sent_id': None,
                    'text': "",
                    'n_token': 0,
                    'raw_lines': []
                }
            else:
                current_sent['raw_lines'].append(line)
                if line_stripped.startswith('# sent_id ='):
                    current_sent['sent_id'] = line_stripped.split('=', 1)[1].strip()
                elif line_stripped.startswith('# text ='):
                    current_sent['text'] = line_stripped.split('=', 1)[1].strip()
                elif not line_stripped.startswith('#'):
                    parts = line_stripped.split('\t')
                    if len(parts) > 0:
                        word_id = parts[0]
                        if word_id.isdigit():
                            current_sent['n_token'] += 1
                            
        if current_sent['sent_id'] is not None:
            current_sent['clean_tokens'] = count_clean_tokens(current_sent['text'])
            sentences.append(current_sent)
            
    return sentences

def main():
    # --- 1. ACCESSO AUTOMATICO ALLA CARTELLA 'postwita' ---
    cartella_input = os.path.join(os.getcwd(), "postwita")
    
    if not os.path.exists(cartella_input):
        print(f"[Errore] Non ho trovato la cartella '{cartella_input}'.")
        print("Assicurati che la cartella 'postwita' sia nella stessa directory di questo script.")
        return
        
    file_conllu = glob.glob(os.path.join(cartella_input, "*.conllu"))
    if not file_conllu:
        print(f"[Errore] Nessun file .conllu trovato dentro la cartella 'postwita'.")
        return
        
    print(f"Trovati {len(file_conllu)} file .conllu nella cartella 'postwita'. Lettura in corso...")

    # --- 2. CARICAMENTO DATI ---
    all_data = []
    for filepath in file_conllu:
        filename = os.path.basename(filepath)
        print(f"[+] Lettura di {filename}...")
        sentences = parse_conllu(filepath)
        
        for s in sentences:
            all_data.append({
                'sent_id': s['sent_id'],
                'text': s['text'],
                'n_token': s['n_token'],
                'clean_tokens': s['clean_tokens'],
                'raw_lines': s['raw_lines']
            })
            
    df = pd.DataFrame(all_data)
    if df.empty:
        print("Nessuna frase valida estratta dai file.")
        return

    # --- 3. CAMPIONAMENTO CASUALE FINO A ~10.000 TOKEN PULITI ---
    SEED = 123
    target_totale = 10000
    
    # Mescola l'intero dataset dei tweet in modo casuale
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    sampled_rows = []
    cumulative_clean_tokens = 0
    
    for idx, row in df_shuffled.iterrows():
        if row['clean_tokens'] == 0:
            continue
            
        sampled_rows.append(row)
        cumulative_clean_tokens += row['clean_tokens']
        
        # Interrompe l'estrazione appena si raggiungono o superano i 10.000 token puliti
        if cumulative_clean_tokens >= target_totale:
            break
            
    df_sampled = pd.DataFrame(sampled_rows)

    # --- 4. CALCOLO METRICHE RICHIESTE SUL CAMPIONE ---
    testi = df_sampled['text'].fillna("").astype(str).tolist()
    
    n_tweets = len(df_sampled)
    tot_token_conllu = df_sampled['n_token'].sum()
    tot_token_puliti = df_sampled['clean_tokens'].sum()
    
    hashtag_pattern = re.compile(r'#\w+')
    mention_pattern = re.compile(r'@\w+')
    
    n_hashtags = sum(len(hashtag_pattern.findall(t)) for t in testi)
    n_mentions = sum(len(mention_pattern.findall(t)) for t in testi)
    
    # --- 5. STAMPA DEL REPORT ---
    print("\n" + "="*50)
    print("         REPORT DI TESTO SUI TWEET ESTRATTI")
    print("="*50)
    print(f"Tweet totali campionati (frasi):          {n_tweets}")
    print(f"Token originali complessivi (CoNLL-U):    {tot_token_conllu}")
    print(f"Token puliti totali (senza link/punct):   {tot_token_puliti}")
    print(f"Numero totale di Hashtag (#):              {n_hashtags}")
    print(f"Numero totale di Menzioni (@):             {n_mentions}")
    print("="*50 + "\n")

    # --- 6. SALVATAGGIO NUOVO FILE CoNLL-U CON METADATI ---
    root = tk.Tk()
    root.withdraw()
    print("Scegli dove salvare il nuovo file CoNLL-U campionato...")
    percorso_conllu = filedialog.asksaveasfilename(
        title="Salva il CoNLL-U con i metadati",
        defaultextension=".conllu",
        filetypes=[("CoNLL-U files", "*.conllu")],
        initialfile="postwita_campionato.conllu"
    )
    
    if percorso_conllu:
        with open(percorso_conllu, 'w', encoding='utf-8') as out_f:
            for _, row in df_sampled.iterrows():
                # Scrive le righe originali della frase preservando commenti e annotazioni
                out_f.write("".join(row['raw_lines']))
                out_f.write("\n")
        print(f"[OK] Nuovo file CoNLL-U salvato correttamente in: {percorso_conllu}")
    else:
        print("Salvataggio annullato.")

if __name__ == "__main__":
    main()