import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. DATI
# ==========================================
# verbi_totali = tutti i token VERB del corpus (i verbi supporto sono INCLUSI).
# Per il chi-quadro le colonne devono essere disgiunte, quindi si usa
# verbi_supporto contro altri_verbi = verbi_totali - verbi_supporto.
data = {
    "corpus": ["ISDT", "ParlaMint", "PolSo", "Ittok", "Gen", "KIParla"],
    "verbi_supporto": [1005, 327, 1019, 2444, 1425, 832],
    "verbi_totali": [7066, 2040, 4466, 9861, 5395, 2384],
}

df = pd.DataFrame(data).set_index("corpus")

if (df["verbi_supporto"] > df["verbi_totali"]).any():
    raise ValueError("verbi_supporto non può superare verbi_totali: controlla i dati.")

df["altri_verbi"] = df["verbi_totali"] - df["verbi_supporto"]
df["% supporto"] = (df["verbi_supporto"] / df["verbi_totali"] * 100).round(2)

tab = df[["verbi_supporto", "altri_verbi"]]

print("📊 Tabella di contingenza osservata:")
print(df[["verbi_supporto", "altri_verbi", "verbi_totali", "% supporto"]])
print("-" * 60)

# ==========================================
# 2. TEST DEL CHI-QUADRATO
# ==========================================
chi2, p, dof, expected = chi2_contingency(tab)
N = tab.values.sum()
cramer_v = np.sqrt(chi2 / (N * (min(tab.shape) - 1)))

print("📝 RISULTATI DEL TEST:")
print(f"• Chi-quadrato: {chi2:.4f}")
print(f"• Gradi di libertà: {dof}")
print(f"• p-value: {p:.4e}")
print(f"• V di Cramér: {cramer_v:.3f}  (effect size: ~0.1 piccolo, ~0.3 medio, ~0.5 grande)")

alpha = 0.05
if p < alpha:
    print(f"\n⚠️ STATISTICAMENTE SIGNIFICATIVO (p < {alpha}): la quota di verbi supporto "
          "sul totale dei verbi cambia tra i corpora.")
else:
    print(f"\n✅ NON SIGNIFICATIVO (p >= {alpha}).")
print("-" * 60)

# ==========================================
# 3. POST-HOC: RESIDUI STANDARDIZZATI CORRETTI
# ==========================================
# Residui corretti di Haberman: sotto H0 seguono una N(0,1), quindi il confronto
# con ±1.96 è legittimo (con i residui di Pearson (O-E)/sqrt(E) non lo è).
# In una tabella k×2 la colonna "altri_verbi" ha gli stessi residui con segno
# opposto, quindi basta guardare la colonna dei verbi supporto.
row_p = tab.sum(axis=1).values[:, None] / N
col_p = tab.sum(axis=0).values[None, :] / N
adj_res = (tab - expected) / np.sqrt(expected * (1 - row_p) * (1 - col_p))

res_vs = adj_res["verbi_supporto"]
post = pd.DataFrame({
    "% supporto": df["% supporto"],
    "residuo_corretto": res_vs.round(2),
})
post["p"] = 2 * norm.sf(res_vs.abs())
post["p_holm"] = multipletests(post["p"], method="holm")[1]
post["esito"] = np.where(
    post["p_holm"] >= alpha, "in linea con l'atteso",
    np.where(post["residuo_corretto"] > 0, "PIÙ verbi supporto", "MENO verbi supporto"),
)

pd.set_option("display.float_format", lambda x: f"{x:.3g}")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("🔍 POST-HOC (residui corretti, Holm sui 6 corpora):")
print(post)
print("\n💡 Residuo > 0: il corpus ha una quota di verbi supporto più alta dell'atteso;"
      "\n   < 0: più bassa. Conta l'esito dopo la correzione di Holm.")