import os
import csv
from collections import Counter

def crea_lessici_per_corpus(cartella_base, cartella_output, usa_lemma=False):
    """
    Legge le sottocartelle (corpora) in una cartella base e crea un CSV di frequenza per ognuna.
    """
    if not os.path.exists(cartella_base):
        print(f"Errore: La cartella base '{cartella_base}' non esiste.")
        return

    # Crea la cartella per i risultati se non esiste già
    if not os.path.exists(cartella_output):
        os.makedirs(cartella_output)
        print(f"Creata cartella di output: '{cartella_output}'\n")

    corpora_trovati = 0

    # Analizza ogni elemento dentro la cartella principale
    for nome_corpus in os.listdir(cartella_base):
        percorso_corpus = os.path.join(cartella_base, nome_corpus)
        
        # Procedi solo se l'elemento è una sottocartella (ignora eventuali file sciolti)
        if os.path.isdir(percorso_corpus):
            corpora_trovati += 1
            contatore_frequenze = Counter()
            file_conllu_trovati = 0
            
            print(f"Elaborazione del corpus: '{nome_corpus}'...")
            
            # Cerca i file .conllu dentro la sottocartella del corpus
            for nome_file in os.listdir(percorso_corpus):
                if nome_file.endswith(".conllu"):
                    file_conllu_trovati += 1
                    percorso_file = os.path.join(percorso_corpus, nome_file)
                    
                    with open(percorso_file, 'r', encoding='utf-8') as f:
                        for linea in f:
                            linea = linea.strip()
                            if not linea or linea.startswith('#'): continue
                            
                            colonne = linea.split('\t')
                            
                            if len(colonne) == 10:
                                id_parola = colonne[0]
                                
                                # Ignora token multi-parola o nodi vuoti
                                if '-' not in id_parola and '.' not in id_parola:
                                    # Scegli colonna (1 per Forma, 2 per Lemma)
                                    parola = colonne[2].lower() if usa_lemma else colonne[1].lower()
                                    contatore_frequenze[parola] += 1
            
            # Se ha trovato e letto dei file nel corpus, salva il CSV
            if file_conllu_trovati > 0:
                nome_csv = f"lessico_{nome_corpus}.csv"
                percorso_csv = os.path.join(cartella_output, nome_csv)
                
                with open(percorso_csv, 'w', encoding='utf-8', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(['Parola', 'Frequenza'])
                    for parola, frequenza in contatore_frequenze.most_common():
                        writer.writerow([parola, frequenza])
                        
                print(f"  -> Successo: elaborati {file_conllu_trovati} file. Salvato in '{nome_csv}'\n")
            else:
                print(f"  -> Attenzione: nessun file .conllu trovato nella cartella '{nome_corpus}'.\n")

    if corpora_trovati == 0:
        print(f"Nessuna sottocartella (corpus) trovata in '{cartella_base}'.")

# --- CONFIGURAZIONE ---
cartella_principale = "conllu_files"
cartella_destinazione = "risultati_lessici"

# Avvia l'elaborazione (usa_lemma=True per i lemmi, False per le forme esatte)
crea_lessici_per_corpus(cartella_principale, cartella_destinazione, usa_lemma=False)