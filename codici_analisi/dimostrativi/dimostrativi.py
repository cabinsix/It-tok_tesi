import os
import csv
from conllu import parse

# --- CONFIGURAZIONE ---
base_path = 'conllu_files'
output_summary = 'analisi_dimostrativi_summary.csv'
output_tokens = 'analisi_dimostrativi_tokens.csv'

exclude_list = ['markit', 'Ittok', 'polso', 'gen']


# --- FUNZIONE ANALISI DIMOSTRATIVI ---
def analyze_demonstratives(sentences, corpus_name):
    results = []

    for s in sentences:
        for t in s:

            # prendiamo DET e PRON
            if t['upos'] not in ['DET', 'PRON']:
                continue

            feats = t['feats'] if t['feats'] else {}

            # filtro dimostrativi UD
            if feats.get('PronType') != 'Dem':
                continue

            lemma = t['lemma'].lower() if t['lemma'] else t['form'].lower()

            results.append({
                "Corpus": corpus_name,
                "UPOS": t['upos'],
                "Lemma": lemma,
                "Forma": t['form']
            })

    return results


# --- CARICAMENTO DATI ---
dataset = {}

for corpus in os.listdir(base_path):
    path = os.path.join(base_path, corpus)

    if not os.path.isdir(path) or corpus in exclude_list:
        continue

    sentences = []

    for file in os.listdir(path):
        if file.endswith('.conllu'):

            file_path = os.path.join(path, file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

            for s in parse(content):

                # filtro ISDT come nel tuo script originale
                if corpus == 'isdt':
                    if "isst_tanl" not in s.metadata.get("sent_id", ""):
                        continue

                sentences.append(s)

    dataset[corpus] = sentences


# --- ANALISI ---
all_stats = []
all_tokens = []

for corpus, sents in dataset.items():

    demo_list = analyze_demonstratives(sents, corpus)

    total = len(demo_list)
    if total == 0:
        continue

    # conteggi DET vs PRON
    det_count = sum(1 for d in demo_list if d['UPOS'] == 'DET')
    pron_count = sum(1 for d in demo_list if d['UPOS'] == 'PRON')

    row = {
        "Corpus": corpus,
        "Tot_Dimostrativi": total,
        "DET_count": det_count,
        "DET_perc": f"{(det_count/total*100):.2f}%",
        "PRON_count": pron_count,
        "PRON_perc": f"{(pron_count/total*100):.2f}%"
    }

    all_stats.append(row)
    all_tokens.extend(demo_list)


# --- SALVATAGGIO CSV SUMMARY ---
if all_stats:
    with open(output_summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"✅ Summary salvato: {output_summary}")
else:
    print("⚠️ Nessun dimostrativo trovato.")


# --- SALVATAGGIO CSV TOKEN ---
if all_tokens:
    with open(output_tokens, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_tokens[0].keys())
        writer.writeheader()
        writer.writerows(all_tokens)

    print(f"✅ Token salvati: {output_tokens}")
