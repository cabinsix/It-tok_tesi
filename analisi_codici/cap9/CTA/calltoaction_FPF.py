import os
import csv
from conllu import parse

# --- CONFIGURAZIONE ---
base_path = 'conllu_files'
output_csv = 'cta_sentence_level.csv'
examples_csv = 'cta_examples.csv' 
exclude_list = ['markit', 'Ittok', 'polso', 'gen']

# metadati fissi per corpus
CORPUS_METADATA = {
    "parlamint": {
        "cst": "presenza",
        "direzionalita": "bidirezionale",
        "simmetria": "asimmetrico"
    },
    "isdt": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "kiparla": {
        "cst": "presenza",
        "direzionalita": "bidirezionale",
        "simmetria": "simmetrico"
    }
}

def is_vocative(token):
    """Identifica se il token è un vocativo basandosi su deprel o feature morfologiche (Case)."""
    deprel = token.get('deprel') or ''
    if 'vocative' in deprel.lower():
        return True
    
    feats = token.get('feats') or {}
    if feats:
        case_feat = feats.get('Case') or ''
        if isinstance(case_feat, str) and case_feat.lower() in ['voc', 'vocative']:
            return True
    return False


def get_cta_type(token, sentence):
    """Identifica CTA (Imperativi, Congiuntivi 3p, Infiniti) con filtri di sicurezza."""
    feats = token.get('feats') or {}
    deprel = token.get('deprel')
    token_id = token.get('id')

    if feats.get('Mood') == 'Imp':
        return "Imperativo"

    if feats.get('Mood') == 'Sub' and feats.get('Person') in ['3', '3p'] and deprel == 'root':
        return "Congiuntivo Esortativo"

    if feats.get('VerbForm') == 'Inf' and deprel == 'root':
        dependents = [t for t in sentence if t.get('head') == token_id]
        deprels_figli = [t.get('deprel') for t in dependents]
        if 'mark' not in deprels_figli and 'aux' not in deprels_figli:
            if "?" not in sentence.metadata.get("text", ""):
                return "Infinito Esortativo"

    return None


def get_doc_id(corpus, filename, sentence):
    """Costruisce doc_id secondo il corpus."""
    if corpus in ["tr-it-tok", "trittok_gen", "trittok_polso", "ittok", "gen", "polso"]:
        return os.path.splitext(filename)[0]

    if corpus == "kiparla":
        return sentence.metadata.get("conversation_id", os.path.splitext(filename)[0])

    return os.path.splitext(filename)[0]


def get_sent_id(corpus, filename, sentence, sent_index):
    """Recupera o costruisce un sent_id univoco."""
    raw_sent_id = sentence.metadata.get("sent_id")
    if raw_sent_id:
        return raw_sent_id
    return f"{os.path.splitext(filename)[0]}_sent_{sent_index:05d}"


results = []
examples_collector = {} # Dizionario per raccogliere esempi: {corpus: [lista_esempi]}

# Variabili per il monitoraggio globale di postwita
totale_vocativi_postwita = 0
totale_vocativi_at_postwita = 0
nome_cartella_postwita_rilevato = None

if not os.path.exists(base_path):
    print(f"Errore: la cartella '{base_path}' non esiste.")
else:
    for corpus in os.listdir(base_path):
        path = os.path.join(base_path, corpus)
        if not os.path.isdir(path) or corpus in exclude_list:
            continue

        print(f"Analisi CTA e Vocativi frase per frase: {corpus}...")
        examples_collector[corpus] = [] 
        
        metadata = CORPUS_METADATA.get(corpus, {"cst": "", "direzionalita": "", "simmetria": ""})

        for filename in os.listdir(path):
            if not filename.endswith('.conllu'): continue
            
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                try: sentences = parse(f.read())
                except: continue

            for sent_index, sent in enumerate(sentences, start=1):
                n_token = 0
                imperativi, infiniti_esort, congiuntivi_esort = 0, 0, 0
                vocativi, vocativi_at = 0, 0
                vocative_words = []
                trovata_cta_nella_frase = None

                for token in sent:
                    if not isinstance(token.get('id'), int) or token.get('upos') == 'PUNCT': 
                        continue
                    n_token += 1

                    # Estrazione dei Vocativi
                    if is_vocative(token):
                        vocativi += 1
                        form = token.get('form') or ''
                        if form:
                            vocative_words.append(form)
                            # Se inizia con @, incrementa il contatore specifico
                            if form.startswith('@'):
                                vocativi_at += 1

                    # Estrazione delle CTA
                    if token.get('upos') in ['VERB', 'AUX']:
                        cta_type = get_cta_type(token, sent)
                        if cta_type:
                            trovata_cta_nella_frase = cta_type
                            if cta_type == "Imperativo": imperativi += 1
                            elif cta_type == "Infinito Esortativo": infiniti_esort += 1
                            elif cta_type == "Congiuntivo Esortativo": congiuntivi_esort += 1

                # Salva statistiche a livello di frase
                results.append({
                    "corpus": corpus,
                    "doc_id": get_doc_id(corpus, filename, sent),
                    "sent_id": get_sent_id(corpus, filename, sent, sent_index),
                    "n_tokens": n_token,
                    "imperativi": imperativi,
                    "infiniti_esort": infiniti_esort,
                    "congiuntivi_esort": congiuntivi_esort,
                    "vocativi": vocativi, 
                    "vocativi_at": vocativi_at,
                    "cst": metadata.get("cst", ""),
                    "direzionalita": metadata.get("direzionalita", ""),
                    "simmetria": metadata.get("simmetria", "")
                })

                # Se la cartella contiene la parola "postwita" (senza distinzione di maiuscole)
                if "postwita" in corpus.lower():
                    nome_cartella_postwita_rilevato = corpus
                    totale_vocativi_postwita += vocativi
                    totale_vocativi_at_postwita += vocativi_at

                # RACCOLTA ESEMPI
                if trovata_cta_nella_frase and len(examples_collector[corpus]) < 5:
                    examples_collector[corpus].append({
                        "corpus": corpus,
                        "cta_type": trovata_cta_nella_frase,
                        "vocatives_found": ", ".join(vocative_words) if vocative_words else "Nessuno",
                        "sentence_text": sent.metadata.get("text", "N/A")
                    })

# --- SALVATAGGIO ---
main_headers = [
    "corpus", "doc_id", "sent_id", "n_tokens", 
    "imperativi", "infiniti_esort", "congiuntivi_esort", 
    "vocativi", "vocativi_at", 
    "cst", "direzionalita", "simmetria"
]

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=main_headers)
    writer.writeheader()
    writer.writerows(results)

with open(examples_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["corpus", "cta_type", "vocatives_found", "sentence_text"])
    writer.writeheader()
    for corpus_data in examples_collector.values():
        writer.writerows(corpus_data)

# --- STAMPA RIEPILOGO DI FINE ANALISI ---
print(f"\n{'='*40}")
print(f"Fatto! Statistiche in {output_csv}, esempi in {examples_csv}.")
print(f"{'='*40}")

if nome_cartella_postwita_rilevato:
    print(f"Rilevato corpus Postwita nella cartella: '{nome_cartella_postwita_rilevato}'")
    print(f"• Vocativi totali rilevati: {totale_vocativi_postwita}")
    print(f"• Di cui con '@' all'inizio: {totale_vocativi_at_postwita}")
    if totale_vocativi_postwita > 0:
        percentuale = (totale_vocativi_at_postwita / totale_vocativi_postwita) * 100
        print(f"• Percentuale di vocativi con '@': {percentuale:.1f}%")
else:
    print("ATTENZIONE: Nessun corpus con la parola 'postwita' trovato nella cartella conllu_files.")