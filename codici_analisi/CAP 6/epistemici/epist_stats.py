import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Dati di input
data = """corpus1,corpus2,U,p
Ittok,gen,963.0,0.1141053779241188
Ittok,isdt,4492.0,9.553583079697579e-17
Ittok,kiparla,499.5,0.13285834453518047
Ittok,parlamint,966.0,0.0009982166865906126
Ittok,polso,1355.5,0.0906550397704306
gen,isdt,2385.0,3.154350621137127e-13
gen,kiparla,316.0,0.7715777322663262
gen,parlamint,551.0,7.531765605140737e-05
gen,polso,809.0,0.0044983551293507895
isdt,kiparla,83.5,3.5389454291867112e-09
isdt,parlamint,379.0,0.0023206460579128507
isdt,polso,304.0,6.941752528497615e-10
kiparla,parlamint,305.5,0.000261876044210887
kiparla,polso,445.5,0.01219199956493481
parlamint,polso,213.0,0.05602018875925973"""

df = pd.read_csv(StringIO(data))
labels = sorted(set(df["corpus1"]).union(df["corpus2"]))
p_mat = pd.DataFrame(np.nan, index=labels, columns=labels)

for _, r in df.iterrows():
    c1, c2, p = r["corpus1"], r["corpus2"], r["p"]
    p_mat.loc[c1, c2] = p
    p_mat.loc[c2, c1] = p

# Diagonale a 0
p_array = p_mat.to_numpy(copy=True)
np.fill_diagonal(p_array, 0)
p_mat.iloc[:, :] = p_array

# --- TRUCCO PER LA SCALA LOGARITMICA ---
# La scala logaritmica non accetta lo 0 (diagonale). 
# Creiamo una copia per il plot dove lo 0 diventa un numero piccolissimo (pari al minimo p-value del dataset)
plot_array = p_mat.to_numpy(copy=True)
plot_array[plot_array == 0] = 1e-17

# Palette Pastello ad alto contrasto: Fucsia Pastello intenso -> Blu Pastello brillante
colors_pastel = ["#FF66CC", "#5D9CEC"]
custom_cmap = LinearSegmentedColormap.from_list("fuchsia_blue", colors_pastel)

plt.figure(figsize=(10, 8))

# Applichiamo LogNorm impostando i limiti da 1e-17 a 1
im = plt.imshow(plot_array, cmap=custom_cmap, norm=LogNorm(vmin=1e-17, vmax=1.0))

# Inserimento dei p-value numerici (usiamo i dati reali di p_mat, non quelli alterati per il plot)
real_array = p_mat.to_numpy()
for i in range(len(labels)):
    for j in range(len(labels)):
        p = real_array[i, j]
        if np.isnan(p):
            continue
        
        # Testo sulla diagonale mostra "0.00", gli altri in notazione scientifica
        text = "0.00" if p == 0 else f"{p:.2e}"
        
        # Colore del testo dinamico per massima leggibilità
        # p piccoli = fucsia (testo nero), p grandi = blu (testo bianco o nero chiaro)
        text_color = "black" if p < 0.001 else "white"
        if p > 0.5: text_color = "black" # per i blu chiarissimi vicino a 1
            
        plt.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.yticks(range(len(labels)), labels)
plt.title("Matrice p-value (Scala Logaritmica - Palette Pastello)", fontsize=14, pad=15)

# Colorbar aggiornata con la scala logaritmica per mostrare gli ordini di grandezza
cbar = plt.colorbar(im, label='p-value (Scala Logaritmica)')
plt.tight_layout()
plt.show()