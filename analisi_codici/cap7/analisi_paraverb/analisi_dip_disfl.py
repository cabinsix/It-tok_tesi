import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAZIONE
# ==========================================
FILE_CSV = "dettaglio_fenomeni_target.csv"
CORPUS_TARGET = "trittok_gen" # Modifica se il nome esatto nel CSV è diverso

sns.set_theme(style="whitegrid", palette="muted")

def analizza_pause_singolo_corpus(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: File '{file_path}' non trovato.")
        return

    if df.empty:
        print("Il dataset è vuoto!")
        return

    # ==========================================
    # 1. FILTRAGGIO DEI DATI
    # ==========================================
    # Filtriamo per corpus esatto e per fenomeno che contiene la parola "pausa"
    df_focus = df[
        (df['corpus'] == CORPUS_TARGET) & 
        (df['fenomeno'].str.contains('pausa', case=False, na=False))
    ]

    if df_focus.empty:
        print(f"Nessuna pausa trovata per il corpus '{CORPUS_TARGET}'.")
        return

    print(f"Analisi delle pause per il corpus: {CORPUS_TARGET}")
    print(f"Totale pause trovate: {len(df_focus)}\n")

    # ==========================================
    # 2. GRAFICI A BARRE (HEAD_POS e HEAD_DEPREL)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Da cosa dipendono le pause nel corpus '{CORPUS_TARGET}'?", fontsize=16)

    # Grafico di Sinistra: Categoria Grammaticale (POS)
    sns.countplot(
        data=df_focus, 
        x="head_pos", 
        hue="fenomeno",
        ax=axes[0],
        order=df_focus['head_pos'].value_counts().index
    )
    axes[0].set_title("Categoria Grammaticale del Nodo Padre (HEAD_POS)")
    axes[0].set_xlabel("Part of Speech (POS)")
    axes[0].set_ylabel("Frequenza Assoluta")
    axes[0].tick_params(axis='x', rotation=45)

    # Grafico di Destra: Relazione Sintattica (DEPREL)
    sns.countplot(
        data=df_focus, 
        x="head_deprel", 
        hue="fenomeno",
        ax=axes[1],
        order=df_focus['head_deprel'].value_counts().index
    )
    axes[1].set_title("Relazione Sintattica del Nodo Padre (HEAD_DEPREL)")
    axes[1].set_xlabel("Dependency Relation (DEPREL)")
    axes[1].set_ylabel("Frequenza Assoluta")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    nome_file_grafico = f"focus_pause_{CORPUS_TARGET}.png"
    plt.savefig(nome_file_grafico, dpi=300)
    print(f" -> Grafico salvato come: {nome_file_grafico}")
    plt.show()

    # ==========================================
    # 3. TABELLE PERCENTUALI IN CONSOLE
    # ==========================================
    print("\n" + "="*50)
    print(" PERCENTUALI PER CATEGORIA GRAMMATICALE (POS)")
    print("="*50)
    # Calcola le percentuali sul totale delle pause
    pos_counts = df_focus['head_pos'].value_counts(normalize=True) * 100
    print(pos_counts.round(1).astype(str) + " %")

    print("\n" + "="*50)
    print(" PERCENTUALI PER RELAZIONE SINTATTICA (DEPREL)")
    print("="*50)
    deprel_counts = df_focus['head_deprel'].value_counts(normalize=True) * 100
    print(deprel_counts.round(1).astype(str) + " %")

    print("\n" + "="*50)
    print(" DETTAGLIO PERCENTUALE PER TIPO DI PAUSA (POS)")
    print("="*50)
    # Ripartizione esatta tra pausa piena e pausa vuota
    pivot_pos = pd.crosstab(df_focus['fenomeno'], df_focus['head_pos'], normalize='index') * 100
    print(pivot_pos.round(1).to_string())

if __name__ == "__main__":
    analizza_pause_singolo_corpus(FILE_CSV)