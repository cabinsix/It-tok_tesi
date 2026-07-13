import os
import csv
from conllu import parse

# --- CONFIGURAZIONE ---
base_path = 'conllu_files'
output_csv = 'analisi_pronomi_persona_numero.csv'

# Filtri: Analizziamo quelli richiesti, saltiamo markit, Ittok, polso, gen
exclude_list = ['markit']

def analyze_pronoun_details(sentences, corpus_name):
    results = []
    
    # Mappatura manuale per fallback se mancano i FEATS
    person_map = {
        'io': ('1', 'Sing'), 'mi': ('1', 'Sing'), 'me': ('1', 'Sing'),
        'noi': ('1', 'Plur'), 'ci': ('1', 'Plur'),
        'tu': ('2', 'Sing'), 'ti': ('2', 'Sing'), 'te': ('2', 'Sing'),
        'voi': ('2', 'Plur'), 'vi': ('2', 'Plur'),
        'lo': ('3', 'Sing'), 'la': ('3', 'Sing'), 'gli': ('3', 'Sing'), 'le': ('3', 'Sing'),
        'li': ('3', 'Plur'), 'loro': ('3', 'Plur'), 'si': ('3', 'Sing/Plur'), 'se': ('3', 'Sing/Plur')
    }

    for s in sentences:
        for t in s:
            if t['upos'] == 'PRON':
                lemma = t['lemma'].lower() if t['lemma'] else t['form'].lower()
                feats = t['feats'] if t['feats'] else {}
                
                # 1. Estrazione Persona e Numero dai tratti UD
                p = feats.get('Person', '3')
                n = feats.get('Number', 'Sing') # Default sensato
                
                # 2. Fallback strategico sui lemmi per precisione 1pl vs 1sg
                if lemma in person_map:
                    p, n = person_map[lemma]
                
                # Creiamo la categoria combinata (es. 1sg, 3pl)
                cat = f"{p}{'sg' if n == 'Sing' else 'pl' if n == 'Plur' else '??'}"
                
                results.append({
                    "Corpus": corpus_name,
                    "Categoria": cat,
                    "Lemma": lemma
                })
    return results

# --- CARICAMENTO E FILTRAGGIO ---
dataset = {}
for corpus in os.listdir(base_path):
    path = os.path.join(base_path, corpus)
    if not os.path.isdir(path) or corpus in exclude_list:
        continue
    
    sentences = []
    for file in os.listdir(path):
        if file.endswith('.conllu'):
            try:
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(os.path.join(path, file), 'r', encoding='latin-1') as f:
                    content = f.read()
            
            for s in parse(content):
                if corpus == 'isdt' and "isst_tanl" not in s.metadata.get("sent_id", ""):
                    continue
                sentences.append(s)
    dataset[corpus] = sentences

# --- ESECUZIONE ANALISI ---
all_stats = []
for corpus, sents in dataset.items():
    pron_list = analyze_pronoun_details(sents, corpus)
    total = len(pron_list)
    if total == 0: continue

    # Conteggi per ogni categoria
    counts = {
        "1sg": sum(1 for p in pron_list if p['Categoria'] == "1sg"),
        "1pl": sum(1 for p in pron_list if p['Categoria'] == "1pl"),
        "2sg": sum(1 for p in pron_list if p['Categoria'] == "2sg"),
        "2pl": sum(1 for p in pron_list if p['Categoria'] == "2pl"),
        "3sg": sum(1 for p in pron_list if p['Categoria'] == "3sg"),
        "3pl": sum(1 for p in pron_list if p['Categoria'] == "3pl")
    }

    row = {"Corpus": corpus, "Tot_Pron": total}
    for cat, count in counts.items():
        row[f"{cat}_count"] = count
        row[f"{cat}_perc"] = f"{(count/total*100):.2f}%"
    
    all_stats.append(row)

# --- SALVATAGGIO CSV ---
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
    writer.writeheader()
    writer.writerows(all_stats)

print(f"Analisi Persona+Numero completata! File: {output_csv}")