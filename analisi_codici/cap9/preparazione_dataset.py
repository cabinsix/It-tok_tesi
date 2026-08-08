# -*- coding: utf-8 -*-
"""
=====================================================================
ESTRATTORE UNIFICATO A LIVELLO DOCUMENTO PER MFA
=====================================================================
Unità di analisi = DOCUMENTO. Ogni riga del CSV = UN documento.
Ogni colonna-feature = misura RELATIVA calcolata come
    (somma numeratori grezzi) / (somma denominatori grezzi)
su TUTTE le frasi del documento — NON media di rapporti per frase.

CHIAVE-DOCUMENTO per corpus:
  - trittok_gen, trittok_polso, kiparla  -> nome del file .conllu (= id_corpus)
  - isdt      -> documento unico, chiave fissa "isst_tanl"
  - parlamint -> documento unico, chiave fissa "parlamint"
  (altri corpora eventuali -> nome file, comportamento di default)

METADATI da metadati.csv (join su id_corpus). 10 colonne categoriali:
  subcorpus, simmetria, direzionalita, CST, distanza, contesto,
  concezione, pianificazione, genere, tipologia_Gen
  (valori .strip() per normalizzare " alta"/"alta", " professionale"/...).

Filtro frase: 5 <= L <= 50 token (no punct). ISDT solo sent_id 'isst_tanl'.
NESSUNA soglia sulla lunghezza del documento (si tengono tutti).

Risorse esterne mancanti -> feature NaN, esecuzione non bloccata.
=====================================================================
"""

import os
import re
import math
import pandas as pd
from collections import defaultdict, Counter
from conllu import parse_incr

# =====================================================================
# 1. CONFIGURAZIONE
# =====================================================================

ROOT_DIR = "conllu_files"
OUTPUT_CSV = "mfa_document_features.csv"
OUTPUT_RAW_CSV = "controllo_grezzo_per_corpus_long.csv"   # conteggi assoluti, formato lungo
OUTPUT_RAW_WIDE_CSV = "controllo_grezzo_per_corpus_wide.csv"  # stesso, formato tabellare

# --- ESCLUSIONE CORPORA -------------------------------------------------
# NB: la cartella KIParla da usare è 'kiparla2.18' (vedi KIPARLA_FOLDER).
# La vecchia cartella 'kiparla' (se presente) va esclusa per non duplicare.
EXCLUDE_CORPORA = {"markit", "Ittok", "gen", "polso", "postwita", "tr-it-tok", "kiparla"}
# ------------------------------------------------------------------------

# --- FILTRO FRASE -------------------------------------------------------
# Solo tetto massimo di lunghezza (come embedding.py): nessun minimo.
MAX_TOKENS = 50
ISDT_FILTER = "isst_tanl"
# ------------------------------------------------------------------------

# --- METADATI DOCUMENTO (metadati.csv) ---------------------------------
META_CSV      = "metadati.csv"
META_SEP      = ";"
META_KEY      = "id_corpus"     # colonna chiave nel CSV (= nome file .conllu / conversazione)
META_COLS     = ["subcorpus", "simmetria", "direzionalita", "CST",
                 "distanza", "contesto", "concezione", "pianificazione",
                 "genere", "tipologia_Gen",
                 "like_count", "view_count", "comment_count"]

# Corpora la cui chiave-documento è il NOME FILE (agganciato a id_corpus).
DOC_BY_FILE   = {"trittok_gen", "trittok_polso"}
# Corpora aggregati in UN UNICO documento (chiave fissa = valore).
DOC_SINGLE    = {"isdt": "isst_tanl", "parlamint": "parlamint"}
# KIParla: cartella 'kiparla2.18'; il DOCUMENTO è la CONVERSAZIONE, presa dal
# sent_id togliendo tutto ciò che segue il primo '_' (es. BOA3017_1 -> BOA3017).
KIPARLA_FOLDER = "kiparla2.18"
# Se il sent_id KIParla avesse un formato diverso, cambia qui la funzione.
# Default: prima parte prima del primo '_'. Es. 'BOA3017_1_tok3' -> 'BOA3017'.
def kiparla_conversation_key(sent_id, filename_key):
    if not sent_id:
        return filename_key
    # tiene solo la parte iniziale prima del primo '_'
    return sent_id.split("_")[0]
# Diagnostica: stampa i primi N sent_id KIParla e le chiavi derivate.
KIPARLA_DEBUG_N = 8
# ------------------------------------------------------------------------

# --- RISORSE ESTERNE (percorsi) ----------------------------------------
WORDNET_FILE    = "IWN-OMW_LMF_v1.0.xml"
VSUPP_CSV       = "pred_analitici_expanded.csv"
VALPAL_CSV      = "database_valpal.csv"
DERIVATARIO_CSV = "derivatario.csv"
NVDB_TXT        = "nvdb.full.txt"
EPISTEMICI_CSV  = "lista_epistemici.csv"
ELITA_CSV       = "ELIta_GOLDEN.csv"
# ------------------------------------------------------------------------

def safe(msg):
    print(msg)

# =====================================================================
# 2. CARICAMENTO RISORSE ESTERNE (fallback silenzioso)
# =====================================================================

UD2IWN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
lemma_measures = {}

def load_wordnet():
    if not os.path.exists(WORDNET_FILE):
        safe(f"  [vaghezza]   risorsa assente ({WORDNET_FILE}) -> feature vuote")
        return
    try:
        from lxml import etree as ET
    except Exception:
        safe("  [vaghezza]   lxml non installato -> feature vuote")
        return
    safe("  [vaghezza]   parsing ItalWordNet...")
    parser = ET.XMLParser(recover=True, huge_tree=True, remove_blank_text=True)
    root = ET.parse(WORDNET_FILE, parser).getroot()
    lemma_synsets = defaultdict(list)
    synset_rel = {}
    for lex in root.findall(".//LexicalEntry"):
        lemma_elem = lex.find("Lemma")
        if lemma_elem is None:
            continue
        lemma = (lemma_elem.get("writtenForm") or "").lower()
        pos = (lemma_elem.get("partOfSpeech") or "").lower()
        for sense in lex.findall("Sense"):
            syn = sense.get("synset")
            if syn:
                lemma_synsets[(lemma, pos)].append(syn)
    for syn in root.findall(".//Synset"):
        sid = syn.get("id")
        synset_rel[sid] = [(r.get("relType"), r.get("target"))
                           for r in syn.findall("SynsetRelation")]
    depth_cache = {}
    def compute_depth(s):
        if s in depth_cache:
            return depth_cache[s]
        hypers = [t for (rt, t) in synset_rel.get(s, []) if rt == "hypernym"]
        if not hypers:
            depth_cache[s] = 0
        else:
            depth_cache[s] = 1 + min(compute_depth(h) for h in hypers if h in synset_rel)
        return depth_cache[s]
    for s in synset_rel:
        compute_depth(s)
    depth_max = max(depth_cache.values()) if depth_cache else 1
    sense_counts = {lp: len(v) for lp, v in lemma_synsets.items()}
    max_senses = max(sense_counts.values()) if sense_counts else 1
    pos_ok = {"n", "v", "a", "r"}
    for (lemma, pos), syns in lemma_synsets.items():
        if pos not in pos_ok:
            continue
        p_norm = (len(syns) - 1) / max_senses if max_senses > 1 else 0
        depths = [depth_cache.get(s, 0) for s in syns]
        avg_depth = sum(depths) / len(depths) if depths else 0
        d_norm = 1 - (avg_depth / depth_max if depth_max > 0 else 0)
        lemma_measures[(lemma, pos)] = {"P": p_norm, "D": d_norm, "V": p_norm * d_norm}
    safe(f"  [vaghezza]   {len(lemma_measures)} lemmi con misure")

vsupp_set = set()
def load_vsupp():
    global vsupp_set
    if not os.path.exists(VSUPP_CSV):
        safe(f"  [vsupp]      risorsa assente ({VSUPP_CSV}) -> feature vuota")
        return
    try:
        df = pd.read_csv(VSUPP_CSV, sep=";")
        vsupp_set = set(df["vsupp"].dropna().str.strip().str.lower())
        safe(f"  [vsupp]      {len(vsupp_set)} verbi supporto")
    except Exception as e:
        safe(f"  [vsupp]      errore lettura ({e}) -> feature vuota")

valpal_db = {}
def _conta_argomenti(frame_text):
    if not frame_text or (isinstance(frame_text, float) and math.isnan(frame_text)):
        return 0
    numeri = re.findall(r"\d+", str(frame_text))
    return max((int(n) for n in numeri), default=0)

def load_valpal():
    global valpal_db
    if not os.path.exists(VALPAL_CSV):
        safe(f"  [valenze]    risorsa assente ({VALPAL_CSV}) -> feature vuota")
        return
    try:
        df = pd.read_csv(VALPAL_CSV, sep=None, engine="python")
        col_lemma = "lemma" if "lemma" in df.columns else df.columns[0]
        col_frame = "valency" if "valency" in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            lemma = str(row[col_lemma]).strip().lower()
            n = _conta_argomenti(row[col_frame])
            valpal_db[lemma] = max(valpal_db.get(lemma, 0), n)
        safe(f"  [valenze]    {len(valpal_db)} verbi")
    except Exception as e:
        safe(f"  [valenze]    errore lettura ({e}) -> feature vuota")

deriv_lemmas = set()
def load_derivatario():
    global deriv_lemmas
    if not os.path.exists(DERIVATARIO_CSV):
        safe(f"  [deriv]      risorsa assente ({DERIVATARIO_CSV}) -> feature vuota")
        return
    try:
        df = pd.read_csv(DERIVATARIO_CSV, sep=";", header=None, dtype=str).fillna("")
        deriv_lemmas = set(df[1].str.lower().str.strip()) - {""}
        safe(f"  [deriv]      {len(deriv_lemmas)} lemmi derivati")
    except Exception as e:
        safe(f"  [deriv]      errore lettura ({e}) -> feature vuota")

nvdb_set = set()
def load_nvdb():
    global nvdb_set
    if not os.path.exists(NVDB_TXT):
        safe(f"  [nvdb]       risorsa assente ({NVDB_TXT}) -> feature vuota")
        return
    try:
        with open(NVDB_TXT, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    nvdb_set.add(line.split()[0].lower())
        safe(f"  [nvdb]       {len(nvdb_set)} parole")
    except Exception as e:
        safe(f"  [nvdb]       errore lettura ({e}) -> feature vuota")

epistemici_single = set()
epistemici_multi = []
def load_epistemici():
    if not os.path.exists(EPISTEMICI_CSV):
        safe(f"  [epistemici] risorsa assente ({EPISTEMICI_CSV}) -> feature vuota")
        return
    try:
        df = pd.read_csv(EPISTEMICI_CSV, engine="python")
        df.columns = df.columns.str.strip()
        col = "forme" if "forme" in df.columns else df.columns[0]
        for f in df[col].dropna().str.lower().str.strip():
            parts = f.split()
            if len(parts) == 1:
                epistemici_single.add(parts[0])
            else:
                epistemici_multi.append(parts)
        safe(f"  [epistemici] {len(epistemici_single)} singole + {len(epistemici_multi)} multiword")
    except Exception as e:
        safe(f"  [epistemici] errore lettura ({e}) -> feature vuota")

EMO_COLS = ["gioia", "tristezza", "rabbia", "disgusto",
            "paura", "fiducia", "sorpresa", "aspettativa"]
elita_lex = {}
def load_elita():
    global elita_lex
    if not os.path.exists(ELITA_CSV):
        safe(f"  [elita]      risorsa assente ({ELITA_CSV}) -> feature vuote")
        return
    try:
        df = pd.read_csv(ELITA_CSV)
        if "Origine" in df.columns:
            df = df[df["Origine"].str.lower() != "golden"]
        df[EMO_COLS] = df[EMO_COLS].apply(pd.to_numeric, errors="coerce")
        avg = df.groupby("Parola")[EMO_COLS].mean()
        avg = avg[(avg >= 0.5).any(axis=1)]
        for parola, row in avg.iterrows():
            elita_lex[str(parola).lower()] = row.to_dict()
        safe(f"  [elita]      {len(elita_lex)} parole emotive")
    except Exception as e:
        safe(f"  [elita]      errore lettura ({e}) -> feature vuote")

# ---------------------------------------------------------------------
# METADATI DOCUMENTO: id_corpus -> {col: valore pulito}
# ---------------------------------------------------------------------
doc_meta = {}
def load_doc_meta():
    global doc_meta
    if not os.path.exists(META_CSV):
        safe(f"  [doc_meta]   risorsa assente ({META_CSV}) -> metadati vuoti")
        return
    try:
        df = pd.read_csv(META_CSV, sep=META_SEP, dtype=str, engine="python")
        df.columns = df.columns.str.strip()
        if META_KEY not in df.columns:
            safe(f"  [doc_meta]   colonna chiave '{META_KEY}' assente in {list(df.columns)[:6]}... -> metadati vuoti")
            return
        present = [c for c in META_COLS if c in df.columns]
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            safe(f"  [doc_meta]   ATTENZIONE colonne mancanti: {missing}")
        for _, row in df.iterrows():
            key = str(row[META_KEY]).strip()
            if not key or key.lower() == "nan":
                continue
            doc_meta[key] = {
                c: (str(row[c]).strip() if (c in present and pd.notna(row[c])) else "")
                for c in META_COLS
            }
        safe(f"  [doc_meta]   {len(doc_meta)} documenti mappati (colonne: {present})")
    except Exception as e:
        safe(f"  [doc_meta]   errore lettura ({e}) -> metadati vuoti")

def get_doc_meta(key):
    m = doc_meta.get(key)
    if m is None:
        # match robusto: strip + case-insensitive
        k2 = (key or "").strip()
        m = doc_meta.get(k2)
        if m is None:
            low = k2.lower()
            for dk, dv in doc_meta.items():
                if dk.strip().lower() == low:
                    m = dv
                    break
    if m is None:
        return {c: "" for c in META_COLS}
    return dict(m)

# =====================================================================
# 3. ANALISI PER SINGOLA FRASE  (conteggi grezzi; identica alle versioni precedenti)
# =====================================================================

SUBORDINATE_RELS = {'acl', 'acl:relcl', 'advcl', 'ccomp', 'xcomp', 'csubj', 'csubj:pass'}
SUBORDINATE_RELS_FRASI = {'advcl', 'ccomp', 'xcomp', 'acl', 'acl:relcl', 'csubj'}
REALIZED_ARGS    = {'nsubj', 'nsubj:pass', 'csubj', 'obj', 'iobj', 'ccomp', 'xcomp'}
NP_MODS          = {'amod', 'nmod', 'nummod', 'flat', 'compound', 'appos'}
NP_HEAD_MODS     = {'det', 'amod', 'nmod', 'nummod', 'acl', 'acl:relcl', 'appos', 'compound'}

PERSON_MAP = {
    'io': ('1', 'Sing'), 'mi': ('1', 'Sing'), 'me': ('1', 'Sing'),
    'noi': ('1', 'Plur'), 'ci': ('1', 'Plur'),
    'tu': ('2', 'Sing'), 'ti': ('2', 'Sing'), 'te': ('2', 'Sing'),
    'voi': ('2', 'Plur'), 'vi': ('2', 'Plur'),
    'lo': ('3', 'Sing'), 'la': ('3', 'Sing'), 'gli': ('3', 'Sing'), 'le': ('3', 'Sing'),
    'li': ('3', 'Plur'), 'loro': ('3', 'Plur'),
    'si': ('3', 'Sing/Plur'), 'se': ('3', 'Sing/Plur'),
}

def get_subtree_size(root_id, children_map):
    ids = set(); stack = [root_id]; visited = set()
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur); ids.add(cur)
        for child in children_map.get(cur, []):
            if child['deprel'] not in {'conj', 'punct'}:
                stack.append(child['id'])
    return len(ids)

def clause_depth(children_map, roots):
    stack = [(r, 0) for r in roots]; maxd = 0
    while stack:
        cid, d = stack.pop()
        if d > maxd:
            maxd = d
        for child in children_map.get(cid, []):
            nd = d + 1 if child['deprel'] in SUBORDINATE_RELS else d
            stack.append((child['id'], nd))
    return maxd

def get_subtree_havelka(node_id, children_map):
    subtree = {node_id}; stack = [node_id]; visited = {node_id}
    while stack:
        cur = stack.pop()
        for child in children_map.get(cur, []):
            cid = child['id']
            if cid not in visited:
                visited.add(cid); subtree.add(cid); stack.append(cid)
    return subtree

def analyze_sentence(sentence):
    words = [t for t in sentence if isinstance(t['id'], int)]
    nopunct = [t for t in words if t['deprel'] != 'punct' and t['upos'] != 'PUNCT']
    n_tok = len(nopunct)

    R = defaultdict(float)
    R['_n_tokens'] = n_tok

    # Il filtro di lunghezza si applica SOLO alle misure che dipendono dalla
    # struttura globale della frase (clause_depth, MDD, non-proiettività):
    # una frase troppo lunga è spesso mal-parsata e falserebbe queste misure.
    # Tutte le altre variabili (token/lemma/relazioni) usano TUTTE le frasi.
    is_short = (n_tok <= MAX_TOKENS)

    children_map = defaultdict(list); id_to_token = {}; roots = []
    for t in words:
        id_to_token[t['id']] = t
        h = t['head']
        if h == 0:
            roots.append(t['id'])
        else:
            children_map[h].append(t)

    upos_c = Counter()
    for t in nopunct:
        upos_c[t['upos']] += 1
    for p in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
              "CCONJ", "SCONJ", "AUX", "PROPN", "NUM", "INTJ"]:
        R[f'pos_{p}'] = upos_c.get(p, 0)

    R['_ttr_lemmas'] = [t['lemma'].lower() for t in nopunct
                        if t['lemma'] and t['upos'] != 'PROPN']

    content = upos_c['NOUN'] + upos_c['VERB'] + upos_c['ADJ'] + upos_c['ADV'] + upos_c['PROPN']
    R['_content'] = content
    R['_function'] = n_tok - content

    # vaghezza
    vag_P = vag_D = vag_V = 0.0; vag_n = 0
    for t in nopunct:
        iwn = UD2IWN.get(t['upos'])
        if iwn and t['lemma']:
            m = lemma_measures.get((t['lemma'].lower(), iwn))
            if m:
                vag_P += m['P']; vag_D += m['D']; vag_V += m['V']; vag_n += 1
    R['_vag_P'] = vag_P; R['_vag_D'] = vag_D; R['_vag_V'] = vag_V; R['_vag_n'] = vag_n

    if nvdb_set:
        lemmi_nvdb = [t['lemma'].lower() for t in nopunct
                      if t['lemma'] and t['upos'] not in ('PROPN', 'NUM')]
        R['_nvdb_hits'] = sum(1 for l in lemmi_nvdb if l in nvdb_set)
        R['_nvdb_den'] = len(lemmi_nvdb)

    if deriv_lemmas:
        lemmi_der = [t['lemma'].lower() for t in nopunct
                     if t['lemma'] and t['upos'] in ('NOUN', 'VERB', 'ADJ', 'ADV')]
        R['_deriv_hits'] = sum(1 for l in lemmi_der if l in deriv_lemmas)
        R['_deriv_den'] = len(lemmi_der)

    if epistemici_single or epistemici_multi:
        forms = [(t['form'] or '').lower() for t in nopunct]
        ep = sum(1 for f in forms if f in epistemici_single)
        for seq in epistemici_multi:
            Ln = len(seq)
            for i in range(len(forms) - Ln + 1):
                if forms[i:i+Ln] == seq:
                    ep += 1
        R['_epistemici'] = ep

    if elita_lex:
        emo_sum = {e: 0.0 for e in EMO_COLS}; emo_hits = 0
        for t in nopunct:
            lem = (t['lemma'] or t['form'] or '').lower()
            if lem in elita_lex:
                emo_hits += 1
                for e in EMO_COLS:
                    emo_sum[e] += elita_lex[lem][e]
        R['_emo_hits'] = emo_hits
        for e in EMO_COLS:
            R[f'_emo_{e}'] = emo_sum[e]

    n_sup = n_voc = 0
    for t in nopunct:
        feats = t['feats'] or {}
        if feats.get('Degree') in ('Abs', 'Sup'):
            n_sup += 1
        if t['deprel'] == 'vocative' or (feats.get('Case', '').lower() in ('voc', 'vocative')):
            n_voc += 1
    R['_n_superlativi'] = n_sup; R['_n_vocativi'] = n_voc

    pron_cat = Counter(); n_pron = 0
    for t in nopunct:
        if t['upos'] != 'PRON':
            continue
        n_pron += 1
        lemma = (t['lemma'] or t['form'] or '').lower()
        feats = t['feats'] or {}
        p = feats.get('Person', '3'); num = feats.get('Number', 'Sing')
        if lemma in PERSON_MAP:
            p, num = PERSON_MAP[lemma]
        suf = 'sg' if num == 'Sing' else 'pl' if num == 'Plur' else 'x'
        pron_cat[f'{p}{suf}'] += 1
    R['_n_pron'] = n_pron
    for c in ['1sg', '1pl', '2sg', '2pl', '3sg', '3pl']:
        R[f'_pron_{c}'] = pron_cat.get(c, 0)

    n_dem = dem_det = dem_pron = 0
    for t in nopunct:
        if t['upos'] in ('DET', 'PRON'):
            feats = t['feats'] or {}
            if feats.get('PronType') == 'Dem':
                n_dem += 1
                if t['upos'] == 'DET':
                    dem_det += 1
                else:
                    dem_pron += 1
    R['_n_dem'] = n_dem; R['_dem_det'] = dem_det; R['_dem_pron'] = dem_pron

    R['_np_nom'] = upos_c['NOUN'] + upos_c['PROPN']
    R['_np_pron'] = upos_c['PRON']

    if vsupp_set:
        R['_vsupp'] = sum(1 for t in nopunct if t['upos'] == 'VERB' and (t['lemma'] or '').lower() in vsupp_set)
        R['_n_verbi'] = upos_c['VERB']

    cta = 0
    for t in words:
        if t['upos'] not in ('VERB', 'AUX'):
            continue
        feats = t['feats'] or {}; deprel = t['deprel']
        if feats.get('Mood') == 'Imp':
            cta += 1
        elif feats.get('Mood') == 'Sub' and feats.get('Person') in ('3', '3p') and deprel == 'root':
            cta += 1
        elif feats.get('VerbForm') == 'Inf' and deprel == 'root':
            figli = [c['deprel'] for c in children_map.get(t['id'], [])]
            if 'mark' not in figli and 'aux' not in figli:
                if "?" not in (sentence.metadata.get("text", "") or ""):
                    cta += 1
    R['_n_cta'] = cta

    verbs = [t for t in nopunct if t['upos'] == 'VERB']
    nv = len(verbs)
    fin = past = pres = fut = sub = cond = imp_mood = ger = inf = part = pass_ = 0
    for t in verbs:
        feats = t['feats'] or {}
        aux_feats = []; cur = t['id']
        for _ in range(3):
            found = None
            for c in children_map.get(cur, []):
                if c['upos'] == 'AUX':
                    found = c; break
            if not found:
                break
            aux_feats.append(found['feats'] or {}); cur = found['id']
        vf = feats.get('VerbForm', '')
        if vf == 'Fin': fin += 1
        if vf == 'Ger': ger += 1
        if vf == 'Inf': inf += 1
        if vf == 'Part': part += 1
        aux_fin = None
        for af in aux_feats:
            if af.get('VerbForm') == 'Fin':
                aux_fin = af; break
        tense = feats.get('Tense', '')
        if not tense and aux_fin:
            tense = aux_fin.get('Tense', '')
        mood = feats.get('Mood', '')
        if not mood and aux_fin:
            mood = aux_fin.get('Mood', '')
        if tense == 'Past': past += 1
        elif tense == 'Pres': pres += 1
        elif tense == 'Fut': fut += 1
        if mood == 'Sub': sub += 1
        elif mood == 'Cnd': cond += 1
        elif mood == 'Imp': imp_mood += 1
        if feats.get('Voice') == 'Pass' or t['deprel'].endswith(':pass'):
            pass_ += 1
    R['_vfin'] = fin; R['_vpast'] = past; R['_vpres'] = pres; R['_vfut'] = fut
    R['_vsub'] = sub; R['_vcond'] = cond; R['_vimp'] = imp_mood
    R['_vger'] = ger; R['_vinf'] = inf; R['_vpart'] = part; R['_vpass'] = pass_
    R['_nv'] = nv

    tot_sub = sub_impl = tot_impl = 0
    for t in nopunct:
        if t['upos'] != 'VERB':
            continue
        feats = t['feats'] or {}; vf = feats.get('VerbForm', 'Unknown')
        is_impl = False
        if vf in ('Inf', 'Ger'):
            is_impl = True
        elif vf == 'Part':
            if not any(c['upos'] == 'AUX' for c in children_map.get(t['id'], [])):
                is_impl = True
        if is_impl:
            tot_impl += 1
        if t['deprel'] in SUBORDINATE_RELS_FRASI:
            tot_sub += 1
            if is_impl:
                sub_impl += 1
    R['_tot_sub'] = tot_sub; R['_sub_impl'] = sub_impl; R['_tot_impl'] = tot_impl

    KNOWN_REL = {'cui', 'quale', 'quali', 'chi', 'quanto', 'quanti', 'dove', 'onde'}
    sub_clauses = expl_conn = 0
    for t in words:
        if t['deprel'] in SUBORDINATE_RELS_FRASI:
            sub_clauses += 1
            has_conn = any(
                c['deprel'] == 'mark'
                or (c['lemma'] or '').lower() == 'che'
                or (c['lemma'] or '').lower() in KNOWN_REL
                for c in children_map.get(t['id'], [])
            )
            if has_conn:
                expl_conn += 1
    R['_sub_clauses'] = sub_clauses; R['_expl_conn'] = expl_conn

    # --- misure di FRASE: solo se la frase è entro MAX_TOKENS ---
    if is_short:
        R['_clause_depth_sum'] = clause_depth(children_map, roots)
        R['_clause_depth_n'] = 1
    else:
        R['_clause_depth_sum'] = 0.0
        R['_clause_depth_n'] = 0

    # MDD identica a embeddingMDD.py: MDD della frase, media sulle frasi.
    # Solo frasi entro MAX_TOKENS e con almeno una dipendenza.
    tot_dist = 0; n_dep = 0
    for t in nopunct:
        h = t['head']
        if h == 0:
            continue
        tot_dist += abs(h - t['id']); n_dep += 1
    if is_short and n_dep > 0:
        R['_mdd_frase'] = tot_dist / n_dep
        R['_mdd_frasi'] = 1
    else:
        R['_mdd_frase'] = 0.0
        R['_mdd_frasi'] = 0

    # Non-proiettività: solo su frasi entro MAX_TOKENS.
    nonproj_arcs = 0
    if is_short:
        for tj in nopunct:
            j = tj['id']; i = tj['head']
            if i == 0:
                continue
            subtree_i = get_subtree_havelka(i, children_map)
            start, end = sorted((i, j))
            for v_id in range(start + 1, end):
                tv = id_to_token.get(v_id)
                if tv is None or tv['deprel'] == 'punct':
                    continue
                if v_id not in subtree_i:
                    nonproj_arcs += 1
                    break
        R['_nonproj_arcs'] = nonproj_arcs
        R['_arcs'] = n_tok
    else:
        R['_nonproj_arcs'] = 0
        R['_arcs'] = 0

    if valpal_db:
        sat_sum = 0.0; sat_n = 0
        for t in nopunct:
            if t['upos'] != 'VERB':
                continue
            lemma = (t['lemma'] or '').lower()
            exp = valpal_db.get(lemma, 0)
            if exp == 0:
                continue
            obs = sum(1 for c in children_map.get(t['id'], []) if c['deprel'] in REALIZED_ARGS)
            if obs > exp:
                obs = exp
            sat_sum += obs / exp; sat_n += 1
        R['_sat_sum'] = sat_sum; R['_sat_n'] = sat_n

    sv = vs = ov = vo = 0
    pos_index = {t['id']: idx for idx, t in enumerate(sentence) if isinstance(t['id'], int)}
    for t in nopunct:
        dep = t['deprel']; h = t['head']
        if h == 0 or h not in pos_index or t['id'] not in pos_index:
            continue
        before = pos_index[t['id']] < pos_index[h]
        if dep in ('nsubj', 'csubj'):
            if before: sv += 1
            else: vs += 1
        elif dep in ('obj', 'ccomp'):
            if before: ov += 1
            else: vo += 1
    R['_sv'] = sv; R['_vs'] = vs; R['_ov'] = ov; R['_vo'] = vo

    modN = Nmod = modNmod = 0
    for head in nopunct:
        if head['upos'] not in ('NOUN', 'PROPN', 'PRON'):
            continue
        mb = ma = False
        for c in children_map.get(head['id'], []):
            if c['deprel'] not in NP_HEAD_MODS:
                continue
            if c['id'] < head['id']:
                mb = True
            elif c['id'] > head['id']:
                ma = True
        if mb and ma:
            modNmod += 1
        elif mb:
            modN += 1
        elif ma:
            Nmod += 1
    R['_modN'] = modN; R['_Nmod'] = Nmod; R['_modNmod'] = modNmod

    np_len_sum = 0; np_count = 0; heavy = Counter()
    for t in nopunct:
        if t['upos'] not in ('NOUN', 'PROPN', 'PRON'):
            continue
        ch = children_map.get(t['id'], [])
        if t['upos'] == 'PRON':
            cat = 'PRON'
        else:
            has_det = any(c['deprel'] == 'det' for c in ch)
            has_rel = any(c['deprel'] == 'acl:relcl' for c in ch)
            has_mod = any(c['deprel'] in NP_MODS for c in ch)
            if has_det and has_rel: cat = 'DET_NP_REL'
            elif has_det and has_mod: cat = 'DET_NP_MOD'
            elif has_mod and not has_det: cat = 'NP_MOD'
            elif has_det: cat = 'DET_NP'
            else: cat = 'NP_nudo'
        heavy[cat] += 1
        np_len_sum += get_subtree_size(t['id'], children_map)
        np_count += 1
    R['_np_count'] = np_count; R['_np_len_sum'] = np_len_sum
    for cat in ['PRON', 'DET_NP_REL', 'DET_NP_MOD', 'NP_MOD', 'DET_NP', 'NP_nudo']:
        R[f'_heavy_{cat}'] = heavy.get(cat, 0)

    return R

# =====================================================================
# 4. AGGREGAZIONE A LIVELLO DOCUMENTO
# =====================================================================

SUM_KEYS = [
    '_n_tokens',
    'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'pos_PRON', 'pos_DET',
    'pos_ADP', 'pos_CCONJ', 'pos_SCONJ', 'pos_AUX', 'pos_PROPN', 'pos_NUM', 'pos_INTJ',
    '_content', '_function',
    '_vag_P', '_vag_D', '_vag_V', '_vag_n',
    '_nvdb_hits', '_nvdb_den',
    '_deriv_hits', '_deriv_den',
    '_epistemici',
    '_emo_hits', '_emo_gioia', '_emo_tristezza', '_emo_rabbia', '_emo_disgusto',
    '_emo_paura', '_emo_fiducia', '_emo_sorpresa', '_emo_aspettativa',
    '_n_superlativi', '_n_vocativi',
    '_n_pron', '_pron_1sg', '_pron_1pl', '_pron_2sg', '_pron_2pl', '_pron_3sg', '_pron_3pl',
    '_n_dem', '_dem_det', '_dem_pron',
    '_np_nom', '_np_pron',
    '_vsupp', '_n_verbi',
    '_n_cta',
    '_vfin', '_vpast', '_vpres', '_vfut', '_vsub', '_vcond', '_vimp',
    '_vger', '_vinf', '_vpart', '_vpass', '_nv',
    '_tot_sub', '_sub_impl', '_tot_impl',
    '_sub_clauses', '_expl_conn',
    '_clause_depth_sum', '_clause_depth_n',
    '_mdd_frase', '_mdd_frasi',
    '_nonproj_arcs', '_arcs',
    '_sat_sum', '_sat_n',
    '_sv', '_vs', '_ov', '_vo',
    '_modN', '_Nmod', '_modNmod',
    '_np_count', '_np_len_sum',
    '_heavy_PRON', '_heavy_DET_NP_REL', '_heavy_DET_NP_MOD',
    '_heavy_NP_MOD', '_heavy_DET_NP', '_heavy_NP_nudo',
]

def new_accumulator():
    acc = {k: 0.0 for k in SUM_KEYS}
    acc['_ttr_lemmas'] = []
    acc['_n_frasi'] = 0
    return acc

def add_sentence_to_acc(acc, R):
    for k in SUM_KEYS:
        if k in R:
            acc[k] += R[k]
    acc['_ttr_lemmas'].extend(R.get('_ttr_lemmas', []))
    acc['_n_frasi'] += 1

def ratio(num, den):
    return (num / den) if den else None

def build_doc_row(acc, meta):
    n = acc['_n_tokens']
    row = {}
    row.update(meta)
    row['n_tokens'] = int(n)
    row['n_frasi'] = int(acc['_n_frasi'])

    for p in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
              "CCONJ", "SCONJ", "AUX", "PROPN", "NUM", "INTJ"]:
        row[f'dens_{p}'] = ratio(acc[f'pos_{p}'], n)

    lemmi = acc['_ttr_lemmas']
    row['ttr'] = ratio(len(set(lemmi)), len(lemmi))

    row['content_ratio'] = ratio(acc['_content'], n)
    row['function_ratio'] = ratio(acc['_function'], n)
    row['lexical_density'] = ratio(acc['_content'], n)

    row['vagh_P'] = ratio(acc['_vag_P'], acc['_vag_n'])
    row['vagh_D'] = ratio(acc['_vag_D'], acc['_vag_n'])
    row['vagh_V'] = ratio(acc['_vag_V'], acc['_vag_n'])

    if nvdb_set:
        row['nvdb_ratio'] = ratio(acc['_nvdb_hits'], acc['_nvdb_den'])
    if deriv_lemmas:
        row['deriv_ratio'] = ratio(acc['_deriv_hits'], acc['_deriv_den'])
    if epistemici_single or epistemici_multi:
        row['epistemici_ratio'] = ratio(acc['_epistemici'], n)

    if elita_lex:
        row['emo_token_ratio'] = ratio(acc['_emo_hits'], n)
        for e in EMO_COLS:
            row[f'emo_{e}'] = ratio(acc[f'_emo_{e}'], n)
        pos = (row.get('emo_gioia') or 0) + (row.get('emo_sorpresa') or 0) \
            + (row.get('emo_fiducia') or 0) + (row.get('emo_aspettativa') or 0)
        neg = (row.get('emo_tristezza') or 0) + (row.get('emo_paura') or 0) \
            + (row.get('emo_disgusto') or 0) + (row.get('emo_rabbia') or 0)
        row['emo_positiva'] = pos
        row['emo_negativa'] = neg

    row['superlativi_ratio'] = ratio(acc['_n_superlativi'], n)
    row['vocativi_ratio'] = ratio(acc['_n_vocativi'], n)

    npron = acc['_n_pron']
    row['pron_ratio'] = ratio(npron, n)
    for c in ['1sg', '1pl', '2sg', '2pl', '3sg', '3pl']:
        row[f'pron_{c}'] = ratio(acc[f'_pron_{c}'], npron)
    row['pron_1'] = ratio(acc['_pron_1sg'] + acc['_pron_1pl'], npron)
    row['pron_2'] = ratio(acc['_pron_2sg'] + acc['_pron_2pl'], npron)
    row['pron_3'] = ratio(acc['_pron_3sg'] + acc['_pron_3pl'], npron)

    row['dem_ratio'] = ratio(acc['_n_dem'], n)
    row['dem_det_ratio'] = ratio(acc['_dem_det'], acc['_n_dem'])
    row['dem_pron_ratio'] = ratio(acc['_dem_pron'], acc['_n_dem'])

    np_tot = acc['_np_nom'] + acc['_np_pron']
    row['np_nominale_ratio'] = ratio(acc['_np_nom'], np_tot)
    row['np_pronominale_ratio'] = ratio(acc['_np_pron'], np_tot)

    if vsupp_set:
        row['vsupp_ratio'] = ratio(acc['_vsupp'], acc['_n_verbi'])

    row['cta_ratio'] = ratio(acc['_n_cta'], n)

    nv = acc['_nv']
    row['v_finiti_ratio'] = ratio(acc['_vfin'], nv)
    row['v_passato_ratio'] = ratio(acc['_vpast'], nv)
    row['v_presente_ratio'] = ratio(acc['_vpres'], nv)
    row['v_futuro_ratio'] = ratio(acc['_vfut'], nv)
    row['v_congiuntivo_ratio'] = ratio(acc['_vsub'], nv)
    row['v_condizionale_ratio'] = ratio(acc['_vcond'], nv)
    row['v_imperativo_ratio'] = ratio(acc['_vimp'], nv)
    row['v_gerundio_ratio'] = ratio(acc['_vger'], nv)
    row['v_infinito_ratio'] = ratio(acc['_vinf'], nv)
    row['v_participio_ratio'] = ratio(acc['_vpart'], nv)
    row['v_passivo_ratio'] = ratio(acc['_vpass'], nv)

    row['subordinate_ratio'] = ratio(acc['_tot_sub'], nv)
    row['implicite_su_sub'] = ratio(acc['_sub_impl'], acc['_tot_sub'])
    row['implicite_ratio'] = ratio(acc['_tot_impl'], nv)
    row['connettivi_espliciti_ratio'] = ratio(acc['_expl_conn'], acc['_sub_clauses'])

    row['clause_depth'] = ratio(acc['_clause_depth_sum'], acc['_clause_depth_n'])
    row['mdd'] = ratio(acc['_mdd_frase'], acc['_mdd_frasi'])   # media MDD-per-frase (= embeddingMDD.py)
    row['nonproj_ratio'] = ratio(acc['_nonproj_arcs'], acc['_arcs'])

    if valpal_db:
        _s = ratio(acc['_sat_sum'], acc['_sat_n'])
        row['saturazione'] = (_s * 100) if _s is not None else None

    subj_tot = acc['_sv'] + acc['_vs']; obj_tot = acc['_ov'] + acc['_vo']
    row['SV_ratio'] = ratio(acc['_sv'], subj_tot)
    row['OV_ratio'] = ratio(acc['_ov'], obj_tot)

    npo = acc['_modN'] + acc['_Nmod'] + acc['_modNmod']
    row['ModN_ratio'] = ratio(acc['_modN'], npo)
    row['NMod_ratio'] = ratio(acc['_Nmod'], npo)
    row['ModNMod_ratio'] = ratio(acc['_modNmod'], npo)

    npc = acc['_np_count']
    for cat in ['PRON', 'DET_NP_REL', 'DET_NP_MOD', 'NP_MOD', 'DET_NP', 'NP_nudo']:
        row[f'heavy_{cat}'] = ratio(acc[f'_heavy_{cat}'], npc)
    row['np_len_media'] = ratio(acc['_np_len_sum'], npc)

    return row

# =====================================================================
# 5. MAIN — aggregazione per documento
# =====================================================================

def iter_conllu_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".conllu"):
                yield os.path.join(root, f)

def doc_id_from_path(fp):
    return os.path.splitext(os.path.basename(fp))[0]

def main():
    safe("=" * 60)
    safe("CARICAMENTO RISORSE ESTERNE")
    safe("=" * 60)
    load_wordnet(); load_vsupp(); load_valpal()
    load_derivatario(); load_nvdb(); load_epistemici(); load_elita()
    load_doc_meta()

    safe("\n" + "=" * 60)
    safe("AGGREGAZIONE PER DOCUMENTO")
    safe("=" * 60)

    if not os.path.isdir(ROOT_DIR):
        safe(f"ERRORE: cartella '{ROOT_DIR}' non trovata.")
        return

    # accumulatori: chiave = (corpus, doc_key)
    accs = {}
    doc_keys_order = []
    unmatched = Counter()   # doc senza corrispondenza nei metadati

    # accumulatore GREZZO per corpus (conteggi assoluti, per CSV di controllo)
    corpus_raw = {}
    corpus_order = []

    corpora = sorted(d for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d)))

    for corpus in corpora:
        if corpus in EXCLUDE_CORPORA:
            safe(f"  [skip] {corpus}")
            continue
        cpath = os.path.join(ROOT_DIR, corpus)
        is_isdt = corpus.lower() == "isdt"
        is_kiparla = corpus == KIPARLA_FOLDER
        single_key = DOC_SINGLE.get(corpus.lower())
        kiparla_debug = []   # (sent_id, chiave derivata, match?) per diagnostica
        kiparla_keys_seen = set()   # tutte le chiavi conversazione viste nei .conllu

        for fp in sorted(iter_conllu_files(cpath)):
            file_key = doc_id_from_path(fp)
            try:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    for sentence in parse_incr(fh):
                        sid = sentence.metadata.get("sent_id", "")
                        if is_isdt and ISDT_FILTER not in sid:
                            continue
                        nopunct = [t for t in sentence
                                   if isinstance(t['id'], int)
                                   and t['deprel'] != 'punct' and t['upos'] != 'PUNCT']
                        # Nessun filtro di lunghezza qui: TUTTE le frasi non vuote
                        # entrano. Il tetto MAX_TOKENS agisce SOLO dentro
                        # analyze_sentence, sulle 3 misure di frase (clause_depth,
                        # MDD, non-proiettività).
                        if len(nopunct) == 0:
                            continue

                        # --- chiave documento ---
                        if single_key is not None:
                            doc_key = single_key          # isdt / parlamint: doc unico
                        elif is_kiparla:
                            doc_key = kiparla_conversation_key(sid, file_key)
                            kiparla_keys_seen.add(doc_key)
                            if len(kiparla_debug) < KIPARLA_DEBUG_N:
                                kiparla_debug.append(
                                    (sid, doc_key, doc_key in doc_meta))
                        else:
                            doc_key = file_key            # nome file = id_corpus

                        akey = (corpus, doc_key)
                        if akey not in accs:
                            accs[akey] = new_accumulator()
                            doc_keys_order.append(akey)

                        R = analyze_sentence(sentence)
                        add_sentence_to_acc(accs[akey], R)

                        # accumulo GREZZO a livello corpus
                        if corpus not in corpus_raw:
                            corpus_raw[corpus] = new_accumulator()
                            corpus_order.append(corpus)
                        add_sentence_to_acc(corpus_raw[corpus], R)
            except Exception as e:
                safe(f"    ! errore in {fp}: {e}")

        # diagnostica KIParla: mostra come vengono derivate le chiavi-conversazione
        if is_kiparla:
            safe(f"  [kiparla] esempi sent_id -> chiave conversazione (match nel CSV?):")
            if not kiparla_debug:
                safe("    (nessuna frase KIParla superava i filtri: controlla cartella/filtri)")
            for sid, key, matched in kiparla_debug:
                flag = "OK" if matched else "NO-MATCH"
                safe(f"    '{sid}'  ->  '{key}'   [{flag}]")
            # chiavi KIParla presenti nel CSV per confronto (con match robusto)
            csv_kip = [k for k, v in doc_meta.items()
                       if str(v.get('subcorpus', '')).lower() == 'kiparla']
            safe(f"    conversazioni trovate nei .conllu : {sorted(kiparla_keys_seen)}")
            safe(f"    id_corpus 'kiparla' nel CSV        : {sorted(csv_kip)}")
            # chiavi orfane: viste nei conllu ma senza metadati (anche con match robusto)
            orfane = [k for k in kiparla_keys_seen
                      if all(v == "" for v in get_doc_meta(k).values())]
            if orfane:
                safe(f"    ⚠️  conversazioni SENZA metadati: {sorted(orfane)}")
                safe(f"       (queste chiavi non compaiono tra gli id_corpus del CSV)")
            else:
                safe(f"    ✔ tutte le conversazioni hanno metadati")

    if not accs:
        safe("Nessun documento estratto. Controlla i filtri.")
        return

    # costruzione righe con metadati
    rows = []
    for (corpus, doc_key) in doc_keys_order:
        acc = accs[(corpus, doc_key)]
        md = get_doc_meta(doc_key)
        if all(v == "" for v in md.values()):
            unmatched[corpus] += 1
        meta = {"corpus": corpus, "doc_id": doc_key}
        meta.update(md)
        rows.append(build_doc_row(acc, meta))

    df = pd.DataFrame(rows)
    id_cols = ["corpus", "doc_id"] + META_COLS + ["n_tokens", "n_frasi"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feat_cols]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    safe("\n" + "=" * 60)
    safe(f"FATTO. {len(df)} documenti x {len(feat_cols)} feature")
    safe(f"Salvato in: {OUTPUT_CSV}")
    safe("=" * 60)
    safe("\nDocumenti per corpus:")
    for c, k in df['corpus'].value_counts().sort_index().items():
        safe(f"  {c:16s} {k}")
    if unmatched:
        safe("\n⚠️  Documenti SENZA metadati (id_corpus non trovato in metadati.csv):")
        for c, k in unmatched.items():
            safe(f"  {c:16s} {k}  -> le colonne metadati saranno vuote")
        safe("  Verifica che il nome del file .conllu coincida con id_corpus nel CSV.")
    safe("\nStatistiche lunghezza documenti (token):")
    safe(df.groupby('corpus')['n_tokens'].describe()[['count', 'min', 'mean', 'max']].to_string())

    # =================================================================
    # OUTPUT DI CONTROLLO: CONTEGGI GREZZI AGGREGATI PER CORPUS
    # =================================================================
    write_corpus_raw_csv(corpus_raw, corpus_order)

def write_corpus_raw_csv(corpus_raw, corpus_order):
    """Scrive OUTPUT_RAW_CSV: conteggi GREZZI (assoluti) aggregati per corpus.
    Formato LUNGO: una riga per (corpus, variabile, valore). Serve come
    controllo dei totali rispetto agli script originali. Nessuna ratio qui."""
    if not corpus_raw:
        return

    # tutte le chiavi grezze sommabili + n_frasi (escludo la lista _ttr_lemmas,
    # non sommabile: al suo posto riporto n. tipi e n. token lessicali)
    keys = [k for k in SUM_KEYS]  # già tutte scalari sommabili
    # descrizioni leggibili delle chiavi principali
    DESC = {
        '_n_tokens': 'token totali (no punct)',
        '_content': 'parole di contenuto', '_function': 'parole funzione',
        '_nv': 'verbi (lessicali)', '_n_verbi': 'verbi (per vsupp)',
        '_n_pron': 'pronomi', '_n_dem': 'dimostrativi',
        '_tot_sub': 'subordinate', '_sub_impl': 'subordinate implicite',
        '_tot_impl': 'implicite totali', '_sub_clauses': 'clausole subordinate (connettivi)',
        '_expl_conn': 'subordinate con connettivo esplicito',
        '_nonproj_arcs': 'archi non-proiettivi', '_arcs': 'archi totali',
        '_np_count': 'teste NP', '_np_len_sum': 'somma lunghezze sottoalberi NP',
        '_vsupp': 'occorrenze verbi supporto', '_n_cta': 'call-to-action',
        '_n_superlativi': 'superlativi', '_n_vocativi': 'vocativi',
        '_epistemici': 'marcatori epistemici',
        '_nvdb_hits': 'lemmi nel vocab. di base', '_nvdb_den': 'lemmi (denom. NVdB)',
        '_deriv_hits': 'lemmi derivati', '_deriv_den': 'lemmi (denom. deriv.)',
        '_vag_n': 'lemmi trovati in ItalWordNet',
        '_sat_n': 'verbi valutati (ValPal)',
        '_sv': 'soggetti pre-verbali', '_vs': 'soggetti post-verbali',
        '_ov': 'oggetti pre-verbali', '_vo': 'oggetti post-verbali',
        '_mdd_frasi': 'frasi con dipendenze (MDD)',
        '_clause_depth_n': 'frasi (clause depth)',
    }

    long_rows = []
    for corpus in corpus_order:
        acc = corpus_raw[corpus]
        # metriche derivate utili al controllo
        acc_n_frasi = int(acc.get('_n_frasi', 0))
        lemmi = acc.get('_ttr_lemmas', [])
        long_rows.append({'corpus': corpus, 'variabile': 'n_frasi',
                          'descrizione': 'frasi analizzate', 'valore': acc_n_frasi})
        long_rows.append({'corpus': corpus, 'variabile': 'n_tipi_lessicali',
                          'descrizione': 'tipi lessicali unici (no PROPN)',
                          'valore': len(set(lemmi))})
        long_rows.append({'corpus': corpus, 'variabile': 'n_token_lessicali',
                          'descrizione': 'token lessicali (no PROPN)',
                          'valore': len(lemmi)})
        for k in keys:
            long_rows.append({
                'corpus': corpus,
                'variabile': k.lstrip('_'),
                'descrizione': DESC.get(k, ''),
                'valore': int(acc.get(k, 0)) if float(acc.get(k, 0)).is_integer()
                          else round(acc.get(k, 0), 4),
            })

    raw_df = pd.DataFrame(long_rows)
    raw_df.to_csv(OUTPUT_RAW_CSV, index=False, encoding="utf-8")

    # versione WIDE (corpus in riga, variabili in colonna) — comoda a colpo d'occhio
    wide = raw_df.pivot(index='corpus', columns='variabile', values='valore')
    wide.to_csv(OUTPUT_RAW_WIDE_CSV, encoding="utf-8")

    safe("\n" + "=" * 60)
    safe(f"CONTROLLO GREZZO salvato:")
    safe(f"  formato lungo : {OUTPUT_RAW_CSV}  ({len(raw_df)} righe)")
    safe(f"  formato wide  : {OUTPUT_RAW_WIDE_CSV}  ({wide.shape[0]} corpora x {wide.shape[1]} variabili)")
    safe("=" * 60)

if __name__ == "__main__":
    main()