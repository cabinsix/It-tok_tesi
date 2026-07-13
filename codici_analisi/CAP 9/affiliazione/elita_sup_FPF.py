import os
import pandas as pd
import conllu

# ======================================================
# CONFIG
# ======================================================
CSV_INPUT = "ELIta_GOLDEN.csv"     # risorsa esterna necessaria
CORPORA_DIR = "conllu_files"
OUTPUT_FILE = "elita_sup_sentence_level.csv"

# ======================================================
# METADATI FISSI CORPORA
# ======================================================
CORPUS_METADATA = {
    "polso": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "gen": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "tr-it-tok": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "trittok_gen": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "trittok_polso": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "kiparla": {
        "cst": "presenza",
        "direzionalita": "bidirezionale",
        "simmetria": "simmetrico"
    },
    "isdt": {
        "cst": "assenza",
        "direzionalita": "monodirezionale",
        "simmetria": "asimmetrico"
    },
    "parlamint": {
        "cst": "presenza",
        "direzionalita": "bidirezionale",
        "simmetria": "asimmetrico"
    }
}

# ======================================================
# LESSICO
# ======================================================
def load_elita_refined(path):
    df = pd.read_csv(path)

    # Filtro originale: escludi 'golden'
    df = df[df['Origine'].str.lower() != 'golden']

    col_emozioni = [
        'gioia', 'tristezza', 'rabbia', 'disgusto',
        'paura', 'fiducia', 'sorpresa', 'aspettativa'
    ]

    df[col_emozioni] = df[col_emozioni].apply(pd.to_numeric, errors='coerce')

    # Media per parola
    lexicon_avg = df.groupby('Parola')[col_emozioni].mean()

    # Filtro originale: almeno un'emozione >= 0.5
    return lexicon_avg[(lexicon_avg >= 0.5).any(axis=1)]

# ======================================================
# HELPERS
# ======================================================
def get_doc_id(corpus, filepath, sentence):
    filename = os.path.basename(filepath)

    if corpus in ["tr-it-tok", "trittok_gen", "trittok_polso", "ittok", "gen", "polso"]:
        return os.path.splitext(filename)[0]

    if corpus == "kiparla":
        return sentence.metadata.get("conversation_id", os.path.splitext(filename)[0])

    return os.path.splitext(filename)[0]


def get_unit_id(corpus, filepath, sentence, sent_index):
    sent_id = sentence.metadata.get("sent_id")
    filename = os.path.basename(filepath)

    if sent_id:
        return sent_id

    if corpus == "kiparla":
        conv_id = sentence.metadata.get("conversation_id", os.path.splitext(filename)[0])
        return f"{conv_id}_sent_{sent_index:05d}"

    return f"{os.path.splitext(filename)[0]}_sent_{sent_index:05d}"

# ======================================================
# ANALISI FRASE-PER-FRASE
# ======================================================
def process_corpus_sentence_level(root_dir, lexicon):
    results = []
    target_words = set(lexicon.index.str.lower())
    col_emozioni = lexicon.columns.tolist()

    for subdir, dirs, files in os.walk(root_dir):
        # esclusione originale
        if 'markit' in subdir.lower():
            continue

        corpus_name = os.path.basename(subdir)
        if not corpus_name:
            continue

        metadata = CORPUS_METADATA.get(
            corpus_name,
            {"cst": "", "direzionalita": "", "simmetria": ""}
        )

        for file in sorted(files):
            if not file.endswith('.conllu'):
                continue

            file_path = os.path.join(subdir, file)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for sent_index, sentence in enumerate(conllu.parse_incr(f), start=1):
                        sent_id = sentence.metadata.get('sent_id', '')

                        # filtro ISDT originale
                        if 'isdt' in subdir.lower() and 'isst_tanl' not in sent_id:
                            continue

                        doc_id = get_doc_id(corpus_name, file_path, sentence)
                        unit_id = get_unit_id(corpus_name, file_path, sentence, sent_index)

                        n_token = 0
                        n_emotional_tokens = 0
                        emotional_lemmas = set()
                        n_superlatives = 0
                        n_vocatives = 0

                        emo_sums = [0.0] * len(col_emozioni)

                        for token in sentence:
                            token_id = token.get("id")

                            # salta multiword token e nodi vuoti
                            if not isinstance(token_id, int):
                                continue

                            if token.get('upos') == 'PUNCT':
                                continue

                            n_token += 1

                            lemma = (token.get('lemma') or token.get('form') or "").lower()

                            # feature originali
                            feats = token.get('feats') or {}
                            is_sup = 1 if feats.get('Degree') in ['Abs', 'Sup'] else 0
                            is_voc = 1 if token.get('deprel') == 'vocative' else 0

                            n_superlatives += is_sup
                            n_vocatives += is_voc

                            if lemma in target_words:
                                emo_values = lexicon.loc[lemma].tolist()
                                n_emotional_tokens += 1
                                emotional_lemmas.add(lemma)

                                for idx, val in enumerate(emo_values):
                                    emo_sums[idx] += val

                        row = {
                            "unit_id": unit_id,
                            "doc_id": doc_id,
                            "sent_id": sent_id if sent_id else unit_id,
                            "corpus": corpus_name,
                            "cst": metadata["cst"],
                            "direzionalita": metadata["direzionalita"],
                            "simmetria": metadata["simmetria"],
                            "n_token": n_token,
                            "n_emotional_tokens": n_emotional_tokens,
                            "n_emotional_lemmas": len(emotional_lemmas),
                            "n_superlatives": n_superlatives,
                            "n_vocatives": n_vocatives
                        }

                        # mantengo la logica delle medie emotive, ma a livello frase
                        # nel codice originale era somma su finestra / window_size
                        # qui diventa somma su frase / n_token
                        for idx, emo_name in enumerate(col_emozioni):
                            row[f"avg_{emo_name}"] = (emo_sums[idx] / n_token) if n_token > 0 else 0.0

                        results.append(row)

            except Exception as e:
                print(f"Nota: saltato file {file_path} per errore: {e}")

    return pd.DataFrame(results)

# ======================================================
# AVVIO
# ======================================================
print("1. Caricamento lessico...")
lex_clean = load_elita_refined(CSV_INPUT)

print(f"2. Analisi frase-per-frase su '{CORPORA_DIR}'...")
final_df = process_corpus_sentence_level(CORPORA_DIR, lex_clean)

final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Fine! Risultati salvati in {OUTPUT_FILE}")
print(final_df.head())