import os
import pandas as pd
import conllu

def load_elita_refined(path):
    # Caricamento sicuro del lessico
    df = pd.read_csv(path)
    # Filtro: escludi 'golden'
    df = df[df['Origine'].str.lower() != 'golden']
    
    col_emozioni = ['gioia', 'tristezza', 'rabbia', 'disgusto', 'paura', 'fiducia', 'sorpresa', 'aspettativa']
    df[col_emozioni] = df[col_emozioni].apply(pd.to_numeric, errors='coerce')
    
    # Raggruppamento e media per parola
    lexicon_avg = df.groupby('Parola')[col_emozioni].mean()
    # Filtro: almeno un'emozione >= 0.5
    return lexicon_avg[(lexicon_avg >= 0.5).any(axis=1)]

def process_corpus_sliding_windows(root_dir, lexicon, window_size=1000):
    all_window_results = []
    target_words = set(lexicon.index)
    col_emozioni = lexicon.columns.tolist()

    for subdir, dirs, files in os.walk(root_dir):
        # Esclusione markit
        if 'markit' in subdir.lower():
            continue
            
        corpus_name = os.path.basename(subdir)
        if not corpus_name: continue

        corpus_tokens = []

        for file in sorted(files):
            if file.endswith('.conllu'):
                file_path = os.path.join(subdir, file)
                try:
                    # errors='replace' previene crash su file non UTF-8
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        for sentence in conllu.parse_incr(f):
                            sent_id = sentence.metadata.get('sent_id', '')
                            
                            # Filtro ISDT
                            if 'isdt' in subdir.lower() and 'isst_tanl' not in sent_id:
                                continue
                            
                            for token in sentence:
                                # Escludi punteggiatura
                                if token['upos'] == 'PUNCT':
                                    continue
                                
                                lemma = (token['lemma'] or token['form'] or "").lower()
                                
                                # Estrazione Feature
                                feats = token.get('feats') or {}
                                is_sup = 1 if feats.get('Degree') in ['Abs', 'Sup'] else 0
                                is_voc = 1 if token.get('deprel') == 'vocative' else 0
                                
                                # Emozioni
                                emo_values = [0.0] * len(col_emozioni)
                                is_emotional = 0
                                if lemma in target_words:
                                    emo_values = lexicon.loc[lemma].tolist()
                                    is_emotional = 1
                                
                                # Salvataggio token (con chiave 'lemma' per evitare KeyError)
                                corpus_tokens.append({
                                    'lemma': lemma,
                                    'superlative': is_sup,
                                    'vocative': is_voc,
                                    'is_emotional': is_emotional,
                                    'emozioni': emo_values
                                })
                except Exception as e:
                    print(f"Nota: saltato file {file_path} per errore: {e}")

        # Calcolo finestre mobili
        for i in range(0, len(corpus_tokens), window_size):
            window = corpus_tokens[i : i + window_size]
            if len(window) < window_size: 
                break 

            # Conteggi
            win_superlatives = sum(t['superlative'] for t in window)
            win_vocatives = sum(t['vocative'] for t in window)
            win_emo_tokens = sum(t['is_emotional'] for t in window)
            
            # Conteggio lemmi emozionali unici
            lemmi_emo_unici = set(t['lemma'] for t in window if t['is_emotional'] == 1)
            win_emo_lemmas = len(lemmi_emo_unici)
            
            # Medie emozionali
            win_emo_sums = [0.0] * len(col_emozioni)
            for t in window:
                for idx, val in enumerate(t['emozioni']):
                    win_emo_sums[idx] += val
            
            res = {
                'corpus': corpus_name,
                'window_id': i // window_size,
                'n_emotional_tokens': win_emo_tokens,
                'n_emotional_lemmas': win_emo_lemmas,
                'n_superlatives': win_superlatives,
                'n_vocatives': win_vocatives
            }
            for idx, emo_name in enumerate(col_emozioni):
                res[f'avg_{emo_name}'] = win_emo_sums[idx] / window_size
            
            all_window_results.append(res)

    return pd.DataFrame(all_window_results)

# --- AVVIO PROGRAMMA ---
csv_input = "ELIta_GOLDEN.csv" # Assicurati che il nome sia esatto
cartella_corpora = "conllu_files"

print("1. Caricamento lessico...")
lex_clean = load_elita_refined(csv_input)

print(f"2. Analisi in corso su '{cartella_corpora}'...")
final_df = process_corpus_sliding_windows(cartella_corpora, lex_clean)

# Salvataggio
output_file = "risultati_finali.csv"
final_df.to_csv(output_file, index=False)

print(f"Fine! Risultati salvati in {output_file}")
print(final_df.head())