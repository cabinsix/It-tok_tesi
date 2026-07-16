import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

# =====================================================
# CARICAMENTO DATI
# =====================================================

df = pd.read_csv("mattr_lemmi_finestre500.csv")

# Se vuoi escludere Ittok:
df = pd.read_csv("mattr_lemmi_finestre500.csv")

# Corpora da escludere
exclude = [
    "Ittok",
    "tr-it-tok",
    "trittok_gen",
    "kiparla",
    "trittok_polso"
]

df = df[~df["cartella"].isin(exclude)]

print("Corpora inclusi nell'analisi:")
print(sorted(df["cartella"].unique()))

# =====================================================
# KRUSKAL-WALLIS
# =====================================================

groups = [
    g["mattr"].values
    for _, g in df.groupby("cartella")
]

H, p = kruskal(*groups)

print("=" * 60)
print("KRUSKAL-WALLIS")
print("=" * 60)
print(f"H = {H:.4f}")
print(f"p = {p:.6e}")


N = len(df)
k = df["cartella"].nunique()

epsilon_sq = (H - k + 1) / (N - k)

print(f"Epsilon squared = {epsilon_sq:.4f}")

# =====================================================
# DUNN + BONFERRONI
# =====================================================

if p < 0.05:

    dunn = sp.posthoc_dunn(
        df,
        val_col="mattr",
        group_col="cartella",
        p_adjust="bonferroni"
    )

    print("\n" + "=" * 60)
    print("DUNN POST-HOC (BONFERRONI)")
    print("=" * 60)

    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.4e}"
    )

    print(dunn)

    dunn.to_csv(
        "dunn_bonferroni_mattr.csv",
        encoding="utf-8"
    )

    print("\nSalvato in: dunn_bonferroni_mattr.csv")

    import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# matrice Dunn già calcolata
df = dunn.copy()

# trasformazione: -log10(p)
# attenzione a p = 0 (li sostituiamo con un valore minimo)
df_safe = df.replace(0, 1e-300)
heatmap_data = -np.log10(df_safe)

plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    annot=False,
    cmap="viridis"
)

plt.title("-log10(p-value) (Dunn post-hoc) → intensità delle differenze")
plt.show()