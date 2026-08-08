import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ── 1. Caricamento ────────────────────────────────────────────────
df = pd.read_csv("risultati_relativi.csv")

# ── 2. Variabile supplementare tipologia_Gen ──────────────────────
# chi ha già tipologia_Gen la tiene;
# per i kiparla senza tipologia_Gen si usa il "genere";
mask_kiparla = df["tipologia_Gen"].isna() & (df["subcorpus"] == "kiparla")
df.loc[mask_kiparla, "tipologia_Gen"] = df.loc[mask_kiparla, "genere"]

# eventuali residui ancora vuoti -> subcorpus
df["tipologia_Gen"] = df["tipologia_Gen"].fillna(df["subcorpus"])

print("Raggruppamento finale (tipologia_Gen):")
print(df.groupby(["subcorpus", "tipologia_Gen"]).size())

# ── 3. Definizione dei gruppi (variabili ATTIVE) ──────────────────
grp_frasale = [
    "clause_embedding","n_implicite_globali","n_sub_implicite","n_subordinate",
    "nonproj_perc_archi","nonproj_perc_frasi",
]

grp_npordine = [

    "ord_OV_ratio","ord_SV_ratio",
    "ordmod_NOUN_ModN","ordmod_NOUN_ModNMod","ordmod_NOUN_NMod",
    "ordmod_PRON_ModN","ordmod_PRON_ModNMod","ordmod_PRON_NMod",

]

grp_pesnp = [
    "np_NOBJ_i_DET_NP_REL","np_NOBJ_ii_DET_NP_MOD","np_NOBJ_iii_NP_MOD",
    "np_NOBJ_iv_DET_NP","np_NOBJ_v_NP_Nudo","np_NOBJ_vi_PRON",
    "np_OBJ_i_DET_NP_REL","np_OBJ_ii_DET_NP_MOD","np_OBJ_iii_NP_MOD",
    "np_OBJ_iv_DET_NP","np_OBJ_v_NP_Nudo","np_OBJ_vi_PRON",
    "np_SOGG_i_DET_NP_REL","np_SOGG_ii_DET_NP_MOD","np_SOGG_iii_NP_MOD",
    "np_SOGG_iv_DET_NP","np_SOGG_v_NP_Nudo","np_SOGG_vi_PRON",
]

grp_perspron= [

    "pron_1pl","pron_1sg","pron_2pl","pron_2sg","pron_3pl","pron_3sg",

]

grp_deriv = [
    "deriv_J","deriv_N","deriv_R","deriv_V",
]

grp_compllex = [
    "nvdb_perc","vaghezza_D","vaghezza_P","vaghezza_V","vsupp_count",
    "MATTR_lemmi","dem_det","dem_pron"
]

grp_complsint = [
    "MDD","saturazione_valpal","sn_pronominale",
]

grp_ctaAFF = [
    "cta_congiuntivi","cta_imperativi","cta_infiniti","cta_vocativi",
    "elita_avg_aspettativa","elita_avg_disgusto","elita_avg_fiducia","elita_avg_gioia",
    "elita_avg_paura","elita_avg_rabbia","elita_avg_sorpresa","elita_avg_tristezza",
    "elita_n_emotional","elita_n_superlatives",
]

grp_disfluenze = [
    "dep_det:redup","dep_discourse","dep_discourse:preconj","dep_discourse:redup",
    "dep_reparandum:redup","disf_pause","disf_pause_piene",
    "disf_prolungamenti","disf_reparandum","disf_ripetizioni","disf_segnali_discorsivi",
]

grp_lessico = [
    "upos_ADJ","upos_ADP","upos_ADV","upos_AUX","upos_CCONJ",
    "upos_DET","upos_NOUN","upos_INTJ","upos_PRON",
    "upos_SCONJ","upos_VERB",
]
grp_epistemici=[
    "epistemici_count",
]
# variabili qualitative SUPPLEMENTARI
grp_supp = [
    "tipologia_Gen","corpus","subcorpus","simmetria","direzionalita","CST",
    "distanza","contesto","concezione","pianificazione","genere",
]

groups = {
    "complessita_frasale": grp_frasale,
    "complessita_lex": grp_compllex,
    "complessita_sint":grp_complsint,
    "cta_affiliazione": grp_ctaAFF,
    "disfluenze": grp_disfluenze,
    "lessico": grp_lessico,
    "ordine SN": grp_npordine,
    "pesantezza SN": grp_pesnp,
    "persone pronominali": grp_perspron,    
    "derivati":grp_deriv,
    "epistemici":grp_epistemici,
    "supplementari_qual": grp_supp,   # gruppo qualitativo supplementare
}

# ── 4. Pulizia: tieni solo colonne esistenti, controlla mancanti ──
all_cols = [c for g in groups.values() for c in g]
mancanti = [c for c in all_cols if c not in df.columns]
if mancanti:
    print("ATTENZIONE, colonne non trovate nel CSV:", mancanti)

groups = {g: [c for c in cols if c in df.columns] for g, cols in groups.items()}

# imposta indice leggibile (documento) e seleziona colonne
X = df.set_index("documento")[[c for cols in groups.values() for c in cols]].copy()

# i gruppi numerici -> forza numerico e riempi NA con la media di colonna
num_groups = [g for g in groups if g != "supplementari_qual"]
num_cols = [c for g in num_groups for c in groups[g]]
X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# le qualitative come stringa/categoria
for c in groups["supplementari_qual"]:
    X[c] = X[c].astype(str)

# ── 5. MFA ────────────────────────────────────────────────────────
mfa = prince.MFA(
    n_components=5,
    n_iter=3,
    copy=True,
    check_input=True,
    engine="sklearn",
    random_state=42,
)

mfa = mfa.fit(
    X,
    groups=groups,
    supplementary_groups=["supplementari_qual"],  # gruppo qualitativo supplementare
)

# ── 6. Output ─────────────────────────────────────────────────────
print("\nInerzia spiegata (%):")
print(mfa.eigenvalues_summary)

coord_ind = mfa.row_coordinates(X)
tip = df.set_index("documento")["tipologia_Gen"]
coord_ind = coord_ind.join(tip)

# ── centroidi per tipologia_Gen ───────────────────────────────────
centroidi = coord_ind.groupby("tipologia_Gen")[[0, 1]].mean()
print("\nCentroidi per tipologia_Gen (Dim1, Dim2):")
print(centroidi)

# ── distanze euclidee tra centroidi (Dim1-Dim2) ───────────────────
dist = squareform(pdist(centroidi[[0, 1]].values, metric="euclidean"))
dist_df = pd.DataFrame(dist, index=centroidi.index, columns=centroidi.index).round(3)
print("\nMatrice delle distanze euclidee tra centroidi (Dim1-Dim2):")
print(dist_df)
dist_df.to_csv("mfa_distanze_centroidi.csv")

# ── 10 variabili attive più significative ─────────────────────────
col_coord = mfa.column_coordinates_
print("\nINDEX SAMPLE:", col_coord.index[:5].tolist())
print("NLEVELS:", col_coord.index.nlevels)

# normalizza i nomi delle dimensioni a 0,1,2...
col_coord.columns = list(range(col_coord.shape[1]))

# l'indice è un MultiIndex (gruppo, variabile): prendi il livello "variabile"
if isinstance(col_coord.index, pd.MultiIndex):
    var_names = col_coord.index.get_level_values(-1)   # ultimo livello = nome variabile
else:
    var_names = col_coord.index

active_cols = [c for g in groups if g != "supplementari_qual" for c in groups[g]]
mask = var_names.isin(active_cols)
col_coord_active = col_coord[mask].copy()

# riduci l'indice al solo nome della variabile per le etichette del grafico
col_coord_active.index = var_names[mask]

# "significatività" = norma sul piano Dim1-Dim2
norme = np.sqrt(col_coord_active[0] ** 2 + col_coord_active[1] ** 2)
top10 = norme.sort_values(ascending=False).head(10).index
print("\nTop 10 variabili attive (norma su Dim1-Dim2):")
print(norme.loc[top10].round(3))

# ── grafico: centroidi colorati + frecce delle 10 variabili top ───
fig, ax = plt.subplots(figsize=(10, 8))

# mappa colore per ciascuna modalità di tipologia_Gen
colore_modalita = {
    # kiparla -> azzurro
    "conversazione libera": "deepskyblue",
    "intervista":           "deepskyblue",
    "lezione":              "deepskyblue",
    "ricevimento":          "deepskyblue",
    # social Gen -> rosa
    "risposta":             "hotpink",
    "stitch":               "hotpink",
    "storytime":            "hotpink",
}
alpha_modalita = {
    "Gen":"hotpink",
}


colori = [colore_modalita.get(lab, "grey") for lab in centroidi.index]
alphas = [alpha_modalita.get(lab, 1.0) for lab in centroidi.index]

# scala le frecce sull'estensione dei centroidi per leggibilità
span = max(centroidi[[0, 1]].abs().max().max(), 1e-9)
arr_max = np.sqrt((col_coord_active.loc[top10, 0] ** 2 +
                   col_coord_active.loc[top10, 1] ** 2)).max()
scale = span / arr_max * 0.9

for v in top10:
    x, y = col_coord_active.loc[v, 0] * scale, col_coord_active.loc[v, 1] * scale
    ax.arrow(0, 0, x, y, color="firebrick", alpha=0.6,
             head_width=span*0.02, length_includes_head=True, zorder=2)
    ax.annotate(v, (x, y), fontsize=8, color="firebrick",
                xytext=(3, 3), textcoords="offset points")

# centroide unico "Gen" (media di risposta/stitch/storytime) in trasparenza
gen_mods = ["risposta", "stitch", "storytime"]
gen_presenti = [m for m in gen_mods if m in centroidi.index]
if gen_presenti:
    gen_x, gen_y = centroidi.loc[gen_presenti, [0, 1]].mean()
    ax.scatter(gen_x, gen_y, s=200, c="hotpink", alpha=0.30,
               edgecolors="black", linewidths=0.6, zorder=2)
    ax.annotate("Gen", (gen_x, gen_y), fontsize=11, fontweight="bold",
                color="hotpink", alpha=0.6,
                xytext=(6, 6), textcoords="offset points", zorder=2)

# centroidi colorati sopra le frecce (alpha per punto)
for (lab, (x, y)), col, al in zip(centroidi.iterrows(), colori, alphas):
    ax.scatter(x, y, s=160, c=col, alpha=al,
               edgecolors="black", linewidths=0.6, zorder=3)
    ax.annotate(lab, (x, y), fontsize=10, fontweight="bold", alpha=al,
                xytext=(5, 5), textcoords="offset points", zorder=4)

# legenda per gruppo di subcorpus
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue",
           markeredgecolor="black", markersize=11, label="KIParla"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="hotpink",
           markeredgecolor="black", markersize=11, label="Format in Gen: risposta/stitch/storytime"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="hotpink",
           markeredgecolor="black", markersize=11, alpha=0.3, label="Centroide medio di TrIttok_Gen")
]
ax.legend(handles=legend_elems, title="subcorpus", loc="best", fontsize=9)

ax.axhline(0, color="grey", lw=.5)
ax.axvline(0, color="grey", lw=.5)
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_title("MFA – centroidi per tipologia_Gen + 10 variabili più significative")
plt.tight_layout()
plt.savefig("mfa_centroidi_biplot.png", dpi=150)
print("\nSalvato: mfa_centroidi_biplot.png e mfa_distanze_centroidi.csv")

# ── 7. PCA dei singoli gruppi ─────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

tip = df.set_index("documento")["tipologia_Gen"]

# stessa palette dei centroidi MFA
colore_modalita = {
    "conversazione libera": "deepskyblue", "intervista": "deepskyblue",
    "lezione": "deepskyblue", "ricevimento": "deepskyblue",
    "risposta": "hotpink", "stitch": "hotpink", "storytime": "hotpink",
}

pca_var = {}  # riepilogo varianza spiegata per gruppo

for gname, cols in groups.items():
    if gname == "supplementari_qual" or len(cols) < 2:
        continue

    Xg = X[cols].apply(pd.to_numeric, errors="coerce")
    Xg = Xg.fillna(Xg.mean())
    Xg = Xg.loc[:, Xg.std() > 0]          # scarta colonne costanti
    if Xg.shape[1] < 2:
        continue

    Z = StandardScaler().fit_transform(Xg)
    pca = PCA(n_components=2, random_state=42).fit(Z)
    scores = pca.transform(Z)
    ev = pca.explained_variance_ratio_ * 100
    pca_var[gname] = ev

    load = pca.components_.T            # (n_var, 2) loadings

    fig, ax = plt.subplots(figsize=(9, 7))

    # punti (documenti) colorati per tipologia_Gen
    labs = tip.reindex(Xg.index)
    cols_pt = [colore_modalita.get(l, "grey") for l in labs]
    ax.scatter(scores[:, 0], scores[:, 1], c=cols_pt, s=40,
               edgecolors="black", linewidths=0.4, alpha=0.7, zorder=2)

    # frecce dei loadings scalate sull'estensione dei punti
    span = max(np.abs(scores[:, :2]).max(), 1e-9)
    lmax = np.sqrt((load ** 2).sum(axis=1)).max()
    scale = span / lmax * 0.9
    for (name, (lx, ly)) in zip(Xg.columns, load):
        x, y = lx * scale, ly * scale
        ax.arrow(0, 0, x, y, color="firebrick", alpha=0.6,
                 head_width=span * 0.02, length_includes_head=True, zorder=3)
        ax.annotate(name, (x, y), fontsize=7, color="firebrick",
                    xytext=(3, 3), textcoords="offset points", zorder=4)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue",
               markeredgecolor="black", markersize=10, label="KIParla"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="hotpink",
               markeredgecolor="black", markersize=10, label="TrIttok_Gen"),
    ]
    ax.legend(handles=leg, loc="best", fontsize=8)

    ax.axhline(0, color="grey", lw=.5)
    ax.axvline(0, color="grey", lw=.5)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(f"PCA gruppo: {gname}")
    plt.tight_layout()

    fname = f"pca_{gname.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Salvato: {fname}  (PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%)")

# riepilogo varianza spiegata
print("\nVarianza spiegata per gruppo (PC1, PC2, cumulata):")
riepilogo = pd.DataFrame(
    {g: [ev[0], ev[1], ev[0] + ev[1]] for g, ev in pca_var.items()},
    index=["PC1_%", "PC2_%", "cum_%"],
).T.round(2)
print(riepilogo)
riepilogo.to_csv("pca_gruppi_varianza.csv")

# ── 8. MFA su un sottoinsieme di gruppi ───────────────────────────
# inserisci qui i gruppi da includere (devono esistere in `groups`)
gruppi_scelti = [
    "complessita_lex",
    "complessita_sint",
    "lessico",
    "ordine SN",
    "pesantezza SN",
    "persone pronominali",   
    "derivati",
    "supplementari_qual",
]

# includi sempre il gruppo qualitativo supplementare
if "supplementari_qual" not in gruppi_scelti:
    gruppi_scelti_full = gruppi_scelti + ["supplementari_qual"]
else:
    gruppi_scelti_full = gruppi_scelti

# controllo nomi
ignoti = [g for g in gruppi_scelti_full if g not in groups]
if ignoti:
    raise ValueError(f"Gruppi non riconosciuti: {ignoti}. Disponibili: {list(groups)}")

groups_sub = {g: groups[g] for g in gruppi_scelti_full}
cols_sub = [c for cols in groups_sub.values() for c in cols]
X_sub = X[cols_sub].copy()

mfa_sub = prince.MFA(
    n_components=5, n_iter=3, copy=True, check_input=True,
    engine="sklearn", random_state=42,
)
mfa_sub = mfa_sub.fit(
    X_sub, groups=groups_sub,
    supplementary_groups=["supplementari_qual"],
)

print("\n=== MFA sottoinsieme:", gruppi_scelti, "===")
print("Inerzia spiegata (%):")
print(mfa_sub.eigenvalues_summary)

# ── centroidi + distanze ──────────────────────────────────────────
coord_sub = mfa_sub.row_coordinates(X_sub)
coord_sub = coord_sub.join(df.set_index("documento")["tipologia_Gen"])
centroidi_sub = coord_sub.groupby("tipologia_Gen")[[0, 1]].mean()
print("\nCentroidi (Dim1, Dim2):")
print(centroidi_sub)

dist_sub = squareform(pdist(centroidi_sub[[0, 1]].values, metric="euclidean"))
dist_sub_df = pd.DataFrame(dist_sub, index=centroidi_sub.index,
                           columns=centroidi_sub.index).round(3)
print("\nDistanze euclidee tra centroidi (Dim1-Dim2):")
print(dist_sub_df)
tag = "_".join(g.replace(" ", "-") for g in gruppi_scelti)
dist_sub_df.to_csv(f"mfa_sub_{tag}_distanze.csv")

# ── top 10 variabili attive + biplot ──────────────────────────────
cc = mfa_sub.column_coordinates_
cc.columns = list(range(cc.shape[1]))
var_names = cc.index.get_level_values(-1) if isinstance(cc.index, pd.MultiIndex) else cc.index

active_sub = [c for g in gruppi_scelti for c in groups_sub[g]]
m = var_names.isin(active_sub)
cc_act = cc[m].copy()
cc_act.index = var_names[m]

norme = np.sqrt(cc_act[0] ** 2 + cc_act[1] ** 2)
top10 = norme.sort_values(ascending=False).head(10).index
print("\nTop 10 variabili attive (norma Dim1-Dim2):")
print(norme.loc[top10].round(3))

colore_modalita = {
    "conversazione libera": "deepskyblue", "intervista": "deepskyblue",
    "lezione": "deepskyblue", "ricevimento": "deepskyblue",
    "risposta": "hotpink", "stitch": "hotpink", "storytime": "hotpink",
}

fig, ax = plt.subplots(figsize=(10, 8))
span = max(centroidi_sub[[0, 1]].abs().max().max(), 1e-9)
arr_max = np.sqrt((cc_act.loc[top10, 0] ** 2 + cc_act.loc[top10, 1] ** 2)).max()
scale = span / arr_max * 0.9

for v in top10:
    x, y = cc_act.loc[v, 0] * scale, cc_act.loc[v, 1] * scale
    ax.arrow(0, 0, x, y, color="firebrick", alpha=0.6,
             head_width=span * 0.02, length_includes_head=True, zorder=2)
    ax.annotate(v, (x, y), fontsize=8, color="firebrick",
                xytext=(3, 3), textcoords="offset points")

for lab, (x, y) in centroidi_sub.iterrows():
    col = colore_modalita.get(lab, "grey")
    ax.scatter(x, y, s=160, c=col, edgecolors="black", linewidths=0.6, zorder=3)
    ax.annotate(lab, (x, y), fontsize=10, fontweight="bold",
                xytext=(5, 5), textcoords="offset points", zorder=4)

ax.axhline(0, color="grey", lw=.5)
ax.axvline(0, color="grey", lw=.5)
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_title(f"MFA sottoinsieme ({', '.join(gruppi_scelti)})")
plt.tight_layout()
plt.savefig(f"mfa_sub_{tag}_biplot.png", dpi=150)
plt.close(fig)
print(f"\nSalvato: mfa_sub_{tag}_biplot.png e mfa_sub_{tag}_distanze.csv")