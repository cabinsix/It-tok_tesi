import os

def analizza_corpus_postwita(cartella_path):
    totale_global_tokens = 0
    totale_global_clean = 0
    totale_global_mentions = 0
    totale_global_hashtags = 0
    
    if not os.path.exists(cartella_path):
        print(f"Errore: La cartella '{cartella_path}' non esiste.")
        return

    files = [f for f in os.listdir(cartella_path) if f.endswith('.conllu')]
    if not files:
        print(f"Nessun file .conllu trovato in '{cartella_path}'.")
        return

    print("=== ANALISI DEI FILE CORRENTI ===")
    
    for file_name in sorted(files):
        file_path = os.path.join(cartella_path, file_name)
        
        file_tokens = 0
        file_clean = 0
        file_mentions = 0
        file_hashtags = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 10:
                    continue
                
                # Salta i token multi-parola (es. 1-2) per contare i token sintattici effettivi
                token_id = parts[0]
                if '-' in token_id or '.' in token_id:
                    continue
                    
                form = parts[1]
                upos = parts[3]
                
                file_tokens += 1
                
                is_mention = form.startswith('@')
                is_hashtag = form.startswith('#')
                is_link = form.startswith('http://') or form.startswith('https://') or form.startswith('www.')
                is_punct = (upos == 'PUNCT')
                
                if is_mention:
                    file_mentions += 1
                if is_hashtag:
                    file_hashtags += 1
                    
                if not is_punct and not is_link:
                    file_clean += 1
                    
        print(f"\nFile: {file_name}")
        print(f"  - Token totali: {file_tokens}")
        print(f"  - Senza punteggiatura e link: {file_clean}")
        print(f"  - Menzioni (@): {file_mentions}")
        print(f"  - Hashtag (#): {file_hashtags}")
        
        totale_global_tokens += file_tokens
        totale_global_clean += file_clean
        totale_global_mentions += file_mentions
        totale_global_hashtags += file_hashtags

    print("\n" + "="*30)
    print("=== CONTEGGIO TOTALE CORPUS ===")
    print(f"Token totali: {totale_global_tokens}")
    print(f"Token senza punteggiatura e senza link: {totale_global_clean}")
    print(f"Menzioni totali (@): {totale_global_mentions}")
    print(f"Hashtag totali (#): {totale_global_hashtags}")

# Specifica il percorso della cartella postwita
percorso_corpus = os.path.join("conllu_files", "postwita")
analizza_corpus_postwita(percorso_corpus)