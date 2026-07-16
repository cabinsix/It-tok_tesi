import os
from collections import Counter
from conllu import parse_incr

# Percorso della cartella principale
BASE_DIR = "conllu_files"

def analizza_corpora():
    # Dizionario per memorizzare i risultati di ogni corpus
    risultati_corpora = {}
    
    # Verifica che la cartella esista
    if not os.path.exists(BASE_DIR):
        print(f"Errore: la cartella '{BASE_DIR}' non esiste.")
        return

    # Esplora ogni sottocartella (corpus)
    for corpus_name in os.listdir(BASE_DIR):
        corpus_path = os.path.join(BASE_DIR, corpus_name)
        
        # Ignora i file nella root, considera solo le sottocartelle
        if not os.path.isdir(corpus_path):
            continue
            
        print(f"Analizzando il corpus: {corpus_name}...")
        
        # Inizializza i contatori per questo corpus
        conteggio_token = 0
        conteggio_pos = Counter()
        conteggio_lemmi = Counter()
        conteggio_relazioni = Counter()
        
        # Cerca tutti i file .conllu nella sottocartella
        for filename in os.listdir(corpus_path):
            if filename.endswith(".conllu"):
                file_path = os.path.join(corpus_path, filename)
                
                # Apri il file. parse_incr legge una frase alla volta (ottimo per file grandi)
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for sentence in parse_incr(f):
                        for token in sentence:
                            # Ignora i token multi-parola (es. ID "1-2" in "dello" = "di" + "lo")
                            # Questo evita di contare sia la parola intera che i suoi componenti.
                            if isinstance(token["id"], tuple):
                                continue
                                
                            conteggio_token += 1
                            
                            # Estrai POS tag (UPOS)
                            if token["upos"]:
                                conteggio_pos[token["upos"]] += 1
                                
                            # Estrai il Lemma
                            if token["lemma"]:
                                conteggio_lemmi[token["lemma"]] += 1
                                
                            # Estrai la relazione di dipendenza (deprel)
                            if token["deprel"]:
                                conteggio_relazioni[token["deprel"]] += 1
                                
        # Salva i dati aggregati del corpus
        risultati_corpora[corpus_name] = {
            "token_totali": conteggio_token,
            "pos": conteggio_pos,
            "lemmi": conteggio_lemmi,
            "relazioni": conteggio_relazioni
        }
        
    # Stampa i risultati a schermo
    stampa_risultati(risultati_corpora)

def stampa_risultati(risultati):
    for corpus, dati in risultati.items():
        print(f"\n{'='*40}")
        print(f"RISULTATI PER IL CORPUS: {corpus.upper()}")
        print(f"{'='*40}")
        print(f"• Token Totali: {dati['token_totali']}")
        
        # Mostra la top 10 per ogni categoria usando .most_common()
        print("\n• Top 10 POS Tag:")
        for pos, count in dati["pos"].most_common(10):
            print(f"  - {pos}: {count}")
            
        print("\n• Top 10 Lemmi:")
        for lemma, count in dati["lemmi"].most_common(10):
            print(f"  - {lemma}: {count}")
            
        print("\n• Top 10 Relazioni (Deprel):")
        for rel, count in dati["relazioni"].most_common(10):
            print(f"  - {rel}: {count}")

if __name__ == "__main__":
    analizza_corpora()