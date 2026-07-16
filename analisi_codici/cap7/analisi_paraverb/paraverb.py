import os
import csv
from pathlib import Path
from collections import Counter
from conllu import parse

# ==========================================
# CONFIGURAZIONE PERCORSI OUTPUT
# ==========================================
INPUT_DIR = Path("conllu_files")
OUTPUT_SENTENCES = Path("disfluenze_per_frase.csv")
OUTPUT_CUMULATIVE = Path("disfluenze_cumulato.csv")

# OUTPUT: Dettaglio di ogni singolo fenomeno con POS, DEPREL e dipendenza sintattica (Head)
OUTPUT_DETTAGLIO_FENOMENI = Path("dettaglio_fenomeni_target.csv")

# Output master per i contesti di TUTTI i fenomeni
OUT_FREQ_DM = Path("frequenze_segnali_discorsivi_per_corpus.txt")
OUT_CONTEXT_BEFORE = Path("contesto_PRIMA_di_tutti_i_fenomeni.txt")
OUT_CONTEXT_AFTER = Path("contesto_DOPO_di_tutti_i_fenomeni.txt")

# ==========================================
# INIZIALIZZAZIONE STRUTTURE DATI
# ==========================================
sentence_rows = []
dettaglio_rows = []  # Conterrà il dettaglio di ogni fenomeno per frase e del suo governatore
cumulative_data = {}  # {corpus_name: {metric: count}}
corpus_frequencies = {}  # Struttura dinamica per contenere i contesti di ogni fenomeno

metrics = [
    "ripetizioni",
    "reparandum",
    "pause",
    "prolungamenti",
    "pause_piene",
    "segnali_discorsivi"
]

def update_context(corpus, phenomenon, direction, token_obj):
    """Funzione ausiliaria per registrare la tripletta di contesto nel Counter corretto"""
    if token_obj is None:
        return
    upos = token_obj["upos"] if token_obj["upos"] else "_"
    deprel = token_obj["deprel"] if token_obj["deprel"] else "_"
    form = token_obj["form"].lower() if token_obj["form"] else "_"
    
    corpus_frequencies[corpus][direction][phenomenon][(upos, deprel, form)] += 1

def add_dettaglio_fenomenO(corpus, file_name, sent_id, fenomeno, token_obj, head_obj=None):
    """Funzione ausiliaria per mappare le caratteristiche intrinseche del token disfluente e del suo governatore"""
    if token_obj is None:
        return
    
    # Gestione del token governatore (head)
    if head_obj == "ROOT":
        head_form, head_pos, head_deprel = "ROOT", "ROOT", "ROOT"
    elif head_obj:
        head_form = head_obj["form"].lower() if head_obj["form"] else "_"
        head_pos = head_obj["upos"] if head_obj["upos"] else "_"
        head_deprel = head_obj["deprel"] if head_obj["deprel"] else "_"
    else:
        head_form, head_pos, head_deprel = "_", "_", "_"
    
    dettaglio_rows.append({
        "corpus": corpus,
        "file": file_name,
        "sent_id": sent_id,
        "fenomeno": fenomeno,
        "token_forma": token_obj["form"].lower() if token_obj["form"] else "_",
        "token_pos": token_obj["upos"] if token_obj["upos"] else "_",
        "token_deprel": token_obj["deprel"] if token_obj["deprel"] else "_",
        "head_forma": head_form,
        "head_pos": head_pos,
        "head_deprel": head_deprel
    })

def analyze_corpus():
    if not INPUT_DIR.exists():
        print(f"Errore: La cartella '{INPUT_DIR}' non esiste.")
        return

    # Scansione dei corpora (sottocartelle)
    for corpus_path in INPUT_DIR.iterdir():
        if not corpus_path.is_dir():
            continue
        
        corpus_name = corpus_path.name
        
        # Inizializzazione contatori macro
        if corpus_name not in cumulative_data:
            cumulative_data[corpus_name] = {m: 0 for m in metrics}
            cumulative_data[corpus_name]["total_sentences"] = 0
            cumulative_data[corpus_name]["total_tokens"] = 0
            
        # Inizializzazione dizionari delle frequenze divisi per fenomeno
        if corpus_name not in corpus_frequencies:
            corpus_frequencies[corpus_name] = {
                "dm": Counter(),
                "before": {m: Counter() for m in metrics},
                "after": {m: Counter() for m in metrics}
            }

        # Lettura file .conllu
        for file_path in corpus_path.glob("*.conllu"):
            print(f"Analisi in corso: {corpus_name}/{file_path.name}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
                
            try:
                sentences = parse(data)
            except Exception as e:
                print(f"Errore nel parsing del file {file_path.name}: {e}")
                continue

            for sentence in sentences:
                sent_id = sentence.metadata.get("sent_id", "UNKNOWN")
                text = sentence.metadata.get("text", "")
                
                sent_counts = {m: 0 for m in metrics}
                cumulative_data[corpus_name]["total_sentences"] += 1
                
                # Filtro token reali (no multi-word)
                valid_tokens = [t for t in sentence if isinstance(t["id"], int)]
                num_tokens = len(valid_tokens)
                
                # Dizionario per mappare rapidamente ID -> Token object (per trovare l'head)
                token_by_id = {t["id"]: t for t in valid_tokens}
                
                for i, token in enumerate(valid_tokens):
                    cumulative_data[corpus_name]["total_tokens"] += 1
                    
                    form = token["form"].lower() if token["form"] else ""
                    lemma = token["lemma"].lower() if token["lemma"] else ""
                    upos = token["upos"] if token["upos"] else "_"
                    deprel = token["deprel"].lower() if token["deprel"] else ""
                    misc = token["misc"] if token["misc"] else {}

                    # Identificazione token adiacenti standard
                    prev_tok = valid_tokens[i-1] if i > 0 else None
                    next_tok = valid_tokens[i+1] if i < num_tokens - 1 else None

                    # Identificazione del governatore sintattico (HEAD)
                    head_id = token["head"]
                    if head_id == 0:
                        head_tok = "ROOT"  # È il nodo principale della frase
                    else:
                        head_tok = token_by_id.get(head_id, None)

                    # 1. RIPETIZIONI
                    if i > 0:
                        if form == prev_tok["form"].lower():
                            sent_counts["ripetizioni"] += 1
                            update_context(corpus_name, "ripetizioni", "before", prev_tok)
                            update_context(corpus_name, "ripetizioni", "after", next_tok)
                            add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "ripetizione", token, head_tok)

                    # 2. REPARANDUM
                    if "reparandum" in deprel:
                        sent_counts["reparandum"] += 1
                        update_context(corpus_name, "reparandum", "before", prev_tok)
                        update_context(corpus_name, "reparandum", "after", next_tok)
                        add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "reparandum", token, head_tok)

                    # 3. PAUSE VUOTE (Gestione differenziata token vs feature)
                    is_pause_token = (lemma == "pause")
                    has_pause_after = (misc.get("PauseAfter") == "Yes")

                    if is_pause_token:
                        sent_counts["pause"] += 1
                        update_context(corpus_name, "pause", "before", prev_tok)
                        update_context(corpus_name, "pause", "after", next_tok)
                        add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "pausa_vuota_token", token, head_tok)
                    elif has_pause_after:
                        sent_counts["pause"] += 1
                        update_context(corpus_name, "pause", "before", token)
                        update_context(corpus_name, "pause", "after", next_tok)
                        add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "pausa_vuota_tratto", token, head_tok)

                    # 4. PROLUNGAMENTI (allungamenti)
                    if misc.get("Prolonged") == "Yes":
                        sent_counts["prolungamenti"] += 1
                        update_context(corpus_name, "prolungamenti", "before", prev_tok)
                        update_context(corpus_name, "prolungamenti", "after", next_tok)
                        add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "prolungamento", token, head_tok)

                    # 5. PAUSE PIENE
                    is_pausa_piena = (form in ["ehm", "eeh"]) or (form in ["eh", "mh"] and "discourse" in deprel)
                    
                    if is_pausa_piena:
                        sent_counts["pause_piene"] += 1
                        update_context(corpus_name, "pause_piene", "before", prev_tok)
                        update_context(corpus_name, "pause_piene", "after", next_tok)
                        add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "pausa_piena", token, head_tok)
                    
                    # 6. SEGNALI DISCORSIVI
                    else:
                        has_dm_misc = any("DM" in str(k) or "DM" in str(v) for k, v in misc.items())
                        is_discourse_deprel = "discourse" in deprel
                        
                        if is_discourse_deprel or has_dm_misc:
                            sent_counts["segnali_discorsivi"] += 1
                            corpus_frequencies[corpus_name]["dm"][form] += 1
                            update_context(corpus_name, "segnali_discorsivi", "before", prev_tok)
                            update_context(corpus_name, "segnali_discorsivi", "after", next_tok)
                            add_dettaglio_fenomenO(corpus_name, file_path.name, sent_id, "segnale_discorsivo", token, head_tok)

                # Aggiornamento cumulativo
                for m in metrics:
                    cumulative_data[corpus_name][m] += sent_counts[m]

                # Salvataggio riga per riga (Macro-riepilogo frasi)
                sentence_rows.append({
                    "corpus": corpus_name,
                    "file": file_path.name,
                    "sent_id": sent_id,
                    "text": text,
                    **sent_counts
                })

    # ==========================================
    # SCRITTURA DEI REPORT DI OUTPUT
    # ==========================================
    
    # CSV: Scrittura del file di dettaglio richiesto (con governatore sintattico)
    with open(OUTPUT_DETTAGLIO_FENOMENI, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "corpus", "file", "sent_id", "fenomeno", 
            "token_forma", "token_pos", "token_deprel",
            "head_forma", "head_pos", "head_deprel"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dettaglio_rows)

    # CSV 1: Frasi
    with open(OUTPUT_SENTENCES, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["corpus", "file", "sent_id", "text"] + metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sentence_rows)

    # CSV 2: Cumulato
    with open(OUTPUT_CUMULATIVE, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["corpus", "total_sentences", "total_tokens"] + metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for corpus, data in cumulative_data.items():
            writer.writerow({"corpus": corpus, **data})

    # TXT 1: Forme dei Segnali Discorsivi
    with open(OUT_FREQ_DM, "w", encoding="utf-8") as f:
        f.write("CORPUS\tFORMA\tFREQUENZA\n")
        for corpus, freqs in corpus_frequencies.items():
            for forma, freq in freqs["dm"].most_common():
                f.write(f"{corpus}\t{forma}\t{freq}\n")

    # TXT 2: Master file contesti PRIMA
    with open(OUT_CONTEXT_BEFORE, "w", encoding="utf-8") as f:
        f.write("CORPUS\tFENOMENO\tUPOS\tDEPREL\tFORMA\tFREQUENZA\n")
        for corpus, categories in corpus_frequencies.items():
            for phenom in metrics:
                for (upos, deprel, form), freq in categories["before"][phenom].most_common():
                    f.write(f"{corpus}\t{phenom}\t{upos}\t{deprel}\t{form}\t{freq}\n")

    # TXT 3: Master file contesti DOPO
    with open(OUT_CONTEXT_AFTER, "w", encoding="utf-8") as f:
        f.write("CORPUS\tFENOMENO\tUPOS\tDEPREL\tFORMA\tFREQUENZA\n")
        for corpus, categories in corpus_frequencies.items():
            for phenom in metrics:
                for (upos, deprel, form), freq in categories["after"][phenom].most_common():
                    f.write(f"{corpus}\t{phenom}\t{upos}\t{deprel}\t{form}\t{freq}\n")

    print(f"\nElaborazione globale completata con successo!")
    print(f" -> {OUTPUT_DETTAGLIO_FENOMENI} (Aggiornato: include le informazioni sull'HEAD sintattico)")
    print(f" -> {OUT_CONTEXT_BEFORE} (Contesti precedenti pronti)")
    print(f" -> {OUT_CONTEXT_AFTER} (Contesti successivi pronti)")

if __name__ == "__main__":
    analyze_corpus()