import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# 1. CARICAMENTO DATI
# =========================================================
df = pd.read_csv("all_samples.csv")

# =========================================================
# 2. ESCLUSIONE CORPORA
# =========================================================
exclude = [
    "trittok_polso",
    "trittok_gen",
    "tr-it-tok",
    "kiparla"
]

df = df[~df["corpus"].isin(exclude)]

print("Corpora inclusi:", df["corpus"].unique())

# =========================================================
# 3. NORMALIZZAZIONE
# =========================================================
df["rate"] = df["markers"] / df["tokens"]

# =========================================================
# 4. ORDINE PER MEDIANA
# =========================================================
order = (
    df.groupby("corpus")["rate"]
    .median()
    .sort_values(ascending=False)
    .index
)

print("\nOrdine corpora (mediana rate):")
print(order)

# =========================================================
# 5. VIOLIN PLOT
# =========================================================
plt.figure(figsize=(11, 6))

sns.violinplot(
    data=df,
    x="corpus",
    y="rate",
    order=order,
    inner="box",   # mostra mediana + IQR dentro il violino
    cut=0          # evita estensioni artificiali oltre i dati
)

sns.stripplot(
    data=df,
    x="corpus",
    y="rate",
    order=order,
    color="black",
    alpha=0.25,
    size=2
)

plt.title("Distribuzione marcatori epistemici (violin plot)")
plt.xlabel("Corpus")
plt.ylabel("Markers / token")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()