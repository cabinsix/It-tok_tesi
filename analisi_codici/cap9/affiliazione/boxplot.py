import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CARICA ===
df = pd.read_csv("elita_sup_sentence_level.csv")   # <-- nome del tuo file

# tieni solo i tre corpora d'interesse
corpora = ["kiparla2.18", "trittok_gen", "trittok_polso", "postwita"]
df = df[df["corpus"].isin(corpora)].copy()

# === COSTRUISCI LE VARIABILI (a livello di frase) ===
# conteggi normalizzati sui token della frase (tasso per token)
df = df[df["n_token"] > 0].copy()
df["superlativi"]   = df["n_superlatives"]     / df["n_token"]
df["token_emotivi"] = df["n_emotional_tokens"] / df["n_token"]
df["lemmi_emotivi"] = df["n_emotional_lemmas"] / df["n_token"]
# valutazioni: somma delle medie emotive (coerente con le tabelle)
df["val_positiva"]  = df[["avg_gioia", "avg_sorpresa", "avg_fiducia", "avg_aspettativa"]].sum(axis=1)
df["val_negativa"]  = df[["avg_tristezza", "avg_paura", "avg_disgusto", "avg_rabbia"]].sum(axis=1)

variabili = [
    ("superlativi",   "Superlativi\n(per token)"),
    ("token_emotivi", "Token emotivi\n(per token)"),
    ("lemmi_emotivi", "Lemmi emotivi\n(per token)"),
    ("val_positiva",  "Valutazione positiva"),
    ("val_negativa",  "Valutazione negativa"),
]

etichette = {"kiparla2.18": "KIParla", "trittok_gen": "TrIttok_Gen",
             "trittok_polso": "TrIttok_PolSo", "postwita":"PoSTwita"}
colori = ["#4C72B0", "#DD8452", "#55A868"]

# === FIGURA: 5 pannelli affiancati ===
fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))

for ax, (col, titolo) in zip(axes, variabili):
    dati = [df[df["corpus"] == c][col].values for c in corpora]
    bp = ax.boxplot(dati, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, colore in zip(bp["boxes"], colori):
        patch.set_facecolor(colore); patch.set_alpha(0.55)
    ax.set_xticks(range(1, len(corpora) + 1))
    ax.set_xticklabels([etichette[c] for c in corpora], fontsize=8, rotation=20)
    ax.set_title(titolo, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.suptitle("Distribuzione a livello di frase per corpus", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("boxplot_5var.png", dpi=200, bbox_inches="tight")
print("Salvato: boxplot_5var.png")