import pandas as pd
import matplotlib.pyplot as plt

# ── Carica il CSV ──────────────────────────────────────────────────────────────
df = pd.read_csv("posizione.csv", sep=";")
df.columns = df.columns.str.strip().str.lower()

print("Colonne:", list(df.columns))
print(f"Righe: {len(df)}\n")
print(df.to_string(index=False))
print()

posizioni = sorted(df["posizione"].unique())
fenomeni  = sorted(df["fenomeno"].unique())
corpora   = sorted(df["corpus"].unique())

# ── Statistiche per corpus × fenomeno × posizione ─────────────────────────────
for corp in corpora:
    for fen in fenomeni:
        for pos in posizioni:
            sub = df[(df["corpus"] == corp) & (df["fenomeno"] == fen) & (df["posizione"] == pos)]
            if sub.empty:
                continue
            print("=" * 60)
            print(f"CORPUS: {corp}  |  FENOMENO: {fen}  |  POSIZIONE: {pos}  ({len(sub)} righe)")
            print("=" * 60)
            print("\n  UPOS:")
            print(sub.groupby("upos")["frequenza"].sum().sort_values(ascending=False).to_string())
            print("\n  DEPREL:")
            print(sub.groupby("deprel")["frequenza"].sum().sort_values(ascending=False).to_string())
            print()

# ── Grafici ────────────────────────────────────────────────────────────────────
COLORS = ["#2a78d6","#1baf7a","#eda100","#4a3aa7",
          "#e34948","#e87ba4","#eb6834","#008300","#73726c"]

def barre_h(ax, series, titolo, multi=False):
    if series.empty:
        ax.text(0.5, 0.5, "nessun dato", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
        ax.set_title(titolo, fontweight="bold", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        return
    s = series.sort_values(ascending=True)
    colori = COLORS[:len(s)] if multi else ["#2a78d6"] * len(s)
    bars = ax.barh(s.index.astype(str), s.values, color=colori, height=0.6)
    ax.set_title(titolo, fontweight="bold", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, s.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=8)

# Una figura per ogni corpus × fenomeno
for corp in corpora:
    for fen in fenomeni:
        sub_cf = df[(df["corpus"] == corp) & (df["fenomeno"] == fen)]
        if sub_cf.empty:
            continue

        n_pos = len(posizioni)
        fig, axes = plt.subplots(n_pos, 2, figsize=(14, 4.5 * n_pos))
        fig.suptitle(f"Corpus: {corp}  —  Fenomeno: {fen}",
                     fontsize=13, fontweight="bold", y=1.01)

        if n_pos == 1:
            axes = [axes]

        for i, pos in enumerate(posizioni):
            sub = sub_cf[sub_cf["posizione"] == pos]
            barre_h(axes[i][0],
                    sub.groupby("upos")["frequenza"].sum(),
                    f"UPOS — {pos}", multi=True)
            barre_h(axes[i][1],
                    sub.groupby("deprel")["frequenza"].sum(),
                    f"DEPREL — {pos}")

        plt.tight_layout()
        nome_file = f"output_{corp}_{fen.replace(' ', '_')}.png"
        plt.savefig(nome_file, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato: {nome_file}")
        plt.show()