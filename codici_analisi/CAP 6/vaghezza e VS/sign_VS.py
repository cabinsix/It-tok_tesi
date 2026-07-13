import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ==========================================
# 1. INSERIMENTO DATI (dalla tua tabella)
# ==========================================
# Nota: ho rinominato la terza colonna in "verbi_f" come nell'immagine
data = {
    "corpus": ["ISDT", "ParlaMint", "PolSo", "Ittok", "Gen", "KIParla"],
    "verbi_supporto": [1005, 327, 1019, 2444, 1425, 832],
    "verbi_f": [7066, 2040, 4466, 9861, 5395, 2384],
}

# Creazione del DataFrame
df = pd.DataFrame(data).set_index("corpus")

print("📊 Tabella di Contingenza Osservata:")
print(df)
print("-" * 50)

# ==========================================
# 2. TEST DEL CHI-QUADRATO
# ==========================================
chi2, p, dof, expected = chi2_contingency(df)

print("📝 RISULTATI DEL TEST:")
print(f"• Statistica Chi-Quadrato: {chi2:.4f}")
print(f"• p-value: {p:.4e}")
print(f"• Gradi di libertà: {dof}")

# Interpretazione del p-value
alpha = 0.05
if p < alpha:
    print(
        f"\n⚠️ Risultato: STATISTICAMENTE SIGNIFICATIVO (p < {alpha})."
        "\nLa distribuzione dei verbi di supporto cambia in modo rilevante a seconda del corpus."
    )
else:
    print(
        f"\n✅ Risultato: NON SIGNIFICATIVO (p >= {alpha})."
        "\nLe differenze riscontrate potrebbero essere dovute al caso."
    )

print("-" * 50)

# ==========================================
# 3. POST-HOC: CALCOLO DEI RESIDUI STANDARDIZZATI
# ==========================================
# I residui aiutano a capire DOVE si concentra la significatività.
# Formula: (Osservato - Atteso) / sqrt(Atteso)
residuals = (df - expected) / np.sqrt(expected)

print("🔍 RESIDUI STANDARDIZZATI (Post-Hoc):")
print(residuals.round(2))
print("\n💡 Guida alla lettura dei residui:")
print("• Un valore > 1.96 indica che il corpus ha PIÙ verbi di quel tipo rispetto alla media attesa.")
print(
    "• Un valore < -1.96 indica che il corpus ha MENO verbi di quel tipo rispetto alla media attesa."
)