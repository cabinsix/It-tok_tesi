import matplotlib.pyplot as plt
import pandas as pd

# 1. Caricamento dei dati della tua tabella
data = {
    "Corpus": ["parlamint", "isdt", "trittok_polso", "trittok_gen", "kiparla"],
    "Gerundio": [2.55, 2.48, 2.08, 1.68, 0.92],
    "Infinito": [21.32, 18.53, 19.07, 20.71, 21.37],
    "Participio": [12.75, 10.87, 3.19, 2.86, 1.45],
}

df = pd.DataFrame(data)

# 2. Configurazione dei colori pastello per assomigliare al target
# [parlamint, isdt, trittok_polso, trittok_gen, kiparla]
colors = ["#8ba3c7", "#5cc2a5", "#e0c59e", "#fcd935", "#ff7f50"]

# Lista dei modi verbali (colonne dei subplot)
modi = ["Gerundio", "Infinito", "Participio"]

# 3. Creazione della figura con 3 sottografici affiancati (condividono l'asse Y)
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

# Regoliamo lo spazio tra i grafici
fig.subplots_adjust(wspace=0.15)

for i, modo in enumerate(modi):
    ax = axes[i]

    # Disegna le barre orizzontali
    bars = ax.barh(df["Corpus"], df[modo], color=colors, height=0.6, zorder=3)

    # Invertiamo l'asse Y per avere 'parlamint' in alto come nella foto
    ax.invert_yaxis()

    # Titolo del singolo pannello
    ax.set_title(modo, fontweight="bold", fontsize=12, pad=15)

    # Aggiunta delle etichette con i valori numerici alla fine di ogni barra
    ax.bar_label(
        bars, fmt="%.1f", padding=6, fontweight="bold", color="#333333"
    )

    # Griglia verticale leggera di sfondo
    ax.grid(axis="x", linestyle="-", linewidth=0.5, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)

    # Pulizia dei bordi (spines) per un look minimal
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    # Rimuove i trattini delle label sull'asse Y e X
    ax.tick_params(axis="both", which="both", left=False, bottom=False)

    # Dimensione dei font dei valori sull'asse X
    ax.tick_params(axis="x", labelsize=10, colors="#666666")

# Personalizzazione delle label dell'asse Y (solo sul primo grafico grazie a sharey=True)
axes[0].tick_params(axis="y", labelsize=11)
for label in axes[0].get_yticklabels():
    label.set_fontweight("bold")
    label.set_color("#555555")

# 4. Label globale in basso al centro per l'asse X
fig.text(0.5, 0.02, "Percentuale", ha="center", fontsize=12, fontweight="bold")

# Ottimizzazione degli spazi finali
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.12)

# Mostra il grafico
plt.show()