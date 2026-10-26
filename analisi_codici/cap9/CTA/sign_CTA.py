import pandas as pd
from scipy import stats

# 1. Carica i dati
df = pd.read_csv("cta_sentence_level.csv")

# 2. Filtra solo i tre corpora di interesse
target_corpora = ["trittok_gen", "trittok_polso"]
df_filtered = df[df["corpus"].isin(target_corpora)].copy()

# 3. Creazione delle variabili binarie (Presenza/Assenza = True/False)
df_filtered['has_vocative'] = df_filtered['vocativi'] > 0
df_filtered['has_imperativo'] = df_filtered['imperativi'] > 0
df_filtered['has_infinito_esort'] = df_filtered['infiniti_esort'] > 0
df_filtered['has_congiuntivo_esort'] = df_filtered['congiuntivi_esort'] > 0

# Variabile combinata: la frase ha ALMENO una CTA di qualsiasi tipo?
df_filtered['has_any_cta'] = (
    (df_filtered['imperativi'] > 0) | 
    (df_filtered['infiniti_esort'] > 0) | 
    (df_filtered['congiuntivi_esort'] > 0)
)

# Definizione dei target di analisi
indicatori = {
    "VOCATIVI": "has_vocative",
    "IMPERATIVI": "has_imperativo",
    "INFINITI ESORTATIVI": "has_infinito_esort",
    "CONGIUNTIVI ESORTATIVI": "has_congiuntivo_esort",
    "QUALSIASI CTA (Totale)": "has_any_cta"
}

coppie = [
    ("trittok_gen", "trittok_polso")
]
num_confronti = len(coppie)

print("=========================================================")
print("    ANALISI STATISTICA MULTI-INDICATORE (PRESENZA/ASSENZA)")
print("=========================================================\n")

# Ciclo di analisi per ogni indicatore
for nome_indicatore, colonna in indicatori.items():
    print(f"=== ANALISI PER: {nome_indicatore} ===")
    
    # Crea la tabella di contingenza (Corpus x Presenza/Assenza)
    tabella = pd.crosstab(df_filtered['corpus'], df_filtered[colonna])
    print("\nTabella delle frequenze osservate:")
    print(tabella)
    print("-" * 30)
    
    # Controlla se l'indicatore è presente nei dati prima di lanciare il test
    # (se una colonna ha solo "False" in tutti i corpora, il test fallirebbe)
    if df_filtered[colonna].nunique() < 2:
        print(f"-> Salto il test per {nome_indicatore}: nessun dato o dati identici in tutti i corpora.\n")
        print("=" * 60 + "\n")
        continue

    # 1. Test Globale (Chi-Quadrato)
    chi2, p_global, dof, expected = stats.chi2_contingency(tabella)
    print(f"Test del Chi-Quadrato Globale:")
    print(f"• Statistica Chi2: {chi2:.4f}")
    print(f"• p-value: {p_global:.4e}")
    
    if p_global < 0.05:
        print("-> RISULTATO: Differenza globale SIGNIFICATIVA (p < 0.05).")
        
        
        # 2. Test Post-Hoc
        for c1, c2 in coppie:
            sub_df = df_filtered[df_filtered['corpus'].isin([c1, c2])]
            sub_table = pd.crosstab(sub_df['corpus'], sub_df[colonna])
            
            # Controllo se ci sono abbastanza dati nella sotto-tabella
            if sub_table.shape != (2, 2):
                print(f"  • {c1} vs {c2}: Impossibile calcolare (dati insufficienti)")
                continue
                
            # Verifica se usare Fisher per bassa numerosità
            _, _, _, expected_sub = stats.chi2_contingency(sub_table)
            if (expected_sub < 5).any():
                oddsratio, p_raw = stats.fisher_exact(sub_table)
                test_usato = "Test Esatto di Fisher"
            else:
                _, p_raw, _, _ = stats.chi2_contingency(sub_table)
                test_usato = "Chi-Quadrato"
            
            # Correzione di Bonferroni
            p_corrected = min(p_raw * num_confronti, 1.0)
            sig_label = "SIGNIFICATIVO" if p_corrected < 0.05 else "NON SIGNIFICATIVO"
            
            print(f"  • {c1} vs {c2}:")
            print(f"    - Test usato: {test_usato}")
            print(f"    - p-value corretto: {p_corrected:.4e} ({sig_label})")
            print(f"    - p-value grezzo: {p_raw:.4e}")
    else:
        print("-> RISULTATO: Nessuna differenza globale significativa (p >= 0.05).")
        print("-> Non è necessario fare confronti post-hoc.")
        
    print("\n" + "=" * 60 + "\n")