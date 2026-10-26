
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

# === 1. CARICA ===
df = pd.read_csv("elita_sup_sentence_level.csv")   # <-- nome del tuo file
corpora_da_escludere = ["isdt", "kiparla","Ittok","gen","polso","parlamint","tr-it-tok","postwita"]   # <-- MODIFICA QUI
df = df[~df["corpus"].isin(corpora_da_escludere)]

# quante frasi per corpus
print(">>> Frasi per corpus:")
print(df["corpus"].value_counts(), "\n")

# === CHIAVE DI AGGREGAZIONE ===
# regola generale: si aggrega per doc_id (il video)
# eccezione KIParla: si aggrega per il prefisso di unit_id prima del primo "_"
#   (es. "conv123_sent_00045" -> "conv123"), cioè per conversazione/parlante
df["agg_key"] = df["doc_id"]
mask_kip = df["corpus"] == "kiparla2.18"
df.loc[mask_kip, "agg_key"] = df.loc[mask_kip, "unit_id"].str.split("_").str[0]

print(">>> UNITÀ di aggregazione per corpus:")
print(df.groupby("corpus")["agg_key"].nunique(), "\n")

# === 2. RICOSTRUISCI LE SOMME GREZZE PER FRASE ===
emozioni = ["gioia", "tristezza", "rabbia", "disgusto",
            "paura", "fiducia", "sorpresa", "aspettativa"]
for emo in emozioni:
    df[f"sum_{emo}"] = df[f"avg_{emo}"] * df["n_token"]

# === 3. AGGREGA PER UNITÀ (video per i social, conversazione per KIParla) ===
agg = df.groupby(["corpus", "agg_key"]).agg(
    n_token=("n_token", "sum"),
    n_emotional_tokens=("n_emotional_tokens", "sum"),
    n_emotional_lemmas=("n_emotional_lemmas", "sum"),
    n_superlatives=("n_superlatives", "sum"),
    **{f"sum_{emo}": (f"sum_{emo}", "sum") for emo in emozioni}
).reset_index()

# normalizza come proporzione sui token dell'unità
for col in ["n_emotional_tokens", "n_emotional_lemmas", "n_superlatives"]:
    agg[f"{col}_prop"] = agg[col] / agg["n_token"]

# medie emotive a livello unità = somma punteggi / token totali
for emo in emozioni:
    agg[f"avg_{emo}"] = agg[f"sum_{emo}"] / agg["n_token"]

agg["valutazione_positiva"] = agg[["avg_gioia", "avg_sorpresa",
                                    "avg_fiducia", "avg_aspettativa"]].sum(axis=1)
agg["valutazione_negativa"] = agg[["avg_tristezza", "avg_paura",
                                    "avg_disgusto", "avg_rabbia"]].sum(axis=1)

# === 4. TEST ===
variabili = {
    "Superlativi": "n_superlatives_prop",
    "Token emotivi": "n_emotional_tokens_prop",
    "Lemmi emotivi": "n_emotional_lemmas_prop",
    "Valutazione positiva": "valutazione_positiva",
    "Valutazione negativa": "valutazione_negativa",
}

print(">>> Medie per corpus (a livello unità):")
print(agg.groupby("corpus")[list(variabili.values())].mean().round(4), "\n")

for nome, col in variabili.items():
    gruppi = [g[col].values for _, g in agg.groupby("corpus")]
    H, p = kruskal(*gruppi)
    print(f"\n=== {nome} ===")
    print(f"Kruskal-Wallis: H = {H:.3f}, p = {p:.4f}",
          "  (significativo)" if p < 0.05 else "  (n.s.)")
    if p < 0.05:
        dunn = sp.posthoc_dunn(agg, val_col=col, group_col="corpus", p_adjust="holm")
        print("Confronti a coppie (Dunn, Holm):")
        print(dunn.round(4))