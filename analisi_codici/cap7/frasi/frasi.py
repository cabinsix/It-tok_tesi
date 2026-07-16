import os
import pandas as pd
from conllu import parse_incr

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita"}

# Relazioni che identificano una subordinata
SUBORDINATE_DEPRELS = {
    'advcl',
    'ccomp',
    'xcomp',
    'acl',
    'acl:relcl',
    'csubj'
}


def process_corpora():
    """Generatore di frasi con filtri cartelle e ISDT"""
    print("Elaborazione in corso...")

    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)

        if folder_name in IGNORE_FOLDERS:
            continue

        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):

                        # Tiene solo le frasi ISST del corpus ISDT
                        if folder_name == "isdt":
                            if "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                                continue

                        yield sentence, folder_name


def analyze_subordinate_depth():

    stats = {}

    print("Analisi incidenza implicite su subordinate...")

    for sentence, corpus in process_corpora():

        if corpus not in stats:
            stats[corpus] = {
                'Tot_Verbi': 0,
                'Tot_ROOT': 0,
                'Tot_Subordinate': 0,
                'Sub_Implicite': 0,
                'Tot_Implicite_Globali': 0
            }

        # Mappa dei figli
        children_map = {}

        for token in sentence:
            head = token["head"]
            children_map.setdefault(head, []).append(token)

        for token in sentence:

            if token["upos"] != "VERB":
                continue

            # Conteggio verbi
            stats[corpus]["Tot_Verbi"] += 1

            # Conteggio ROOT
            if token["deprel"] == "root":
                stats[corpus]["Tot_ROOT"] += 1

            deprel = token["deprel"]
            feats = token["feats"] if token["feats"] else {}
            verb_form = feats.get("VerbForm", "Unknown")

            # Verbo implicito?
            is_implicit = False

            if verb_form in ["Inf", "Ger"]:
                is_implicit = True

            elif verb_form == "Part":

                has_aux = False

                for child in children_map.get(token["id"], []):
                    if child["upos"] == "AUX":
                        has_aux = True
                        break

                if not has_aux:
                    is_implicit = True

            if is_implicit:
                stats[corpus]["Tot_Implicite_Globali"] += 1

            # Subordinate
            if deprel in SUBORDINATE_DEPRELS:

                stats[corpus]["Tot_Subordinate"] += 1

                if is_implicit:
                    stats[corpus]["Sub_Implicite"] += 1

    # Costruzione DataFrame
    rows = []

    for corpus, data in stats.items():

        tot_verbi = data["Tot_Verbi"]
        tot_sub = data["Tot_Subordinate"]

        rows.append({
            "Corpus": corpus,
            "N_Verbi": tot_verbi,
            "N_ROOT": data["Tot_ROOT"],
            "N_Subordinate": tot_sub,
            "N_Sub_Implicite": data["Sub_Implicite"],
            "% Subordinate (su tot)": round((tot_sub / tot_verbi) * 100, 1) if tot_verbi else 0,
            "% Implicite (su Sub)": round((data["Sub_Implicite"] / tot_sub) * 100, 1) if tot_sub else 0
        })

    df = pd.DataFrame(rows)

    return df[
        [
            "Corpus",
            "N_Verbi",
            "N_ROOT",
            "N_Subordinate",
            "N_Sub_Implicite",
            "% Subordinate (su tot)",
            "% Implicite (su Sub)"
        ]
    ]


if __name__ == "__main__":

    df = analyze_subordinate_depth()

    print("\n--- RISULTATI ---")
    print(df.to_string(index=False))

    print("\nTotale VERBI:", df["N_Verbi"].sum())
    print("Totale ROOT:", df["N_ROOT"].sum())