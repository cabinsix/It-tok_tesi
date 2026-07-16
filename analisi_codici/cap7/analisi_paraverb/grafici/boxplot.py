import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def genera_dotplot_separato():
    # 1. Inserimento dei dati estratti dall'immagine
    data = {
        "corpus": ["TrIttok_Gen", "KIParla", "TrIttok_PolSo"],
        "ripetizioni": [5.4409431, 13.4368308, 3.7586919],
        "reparandum": [6.12106098, 17.2376874, 4.13456117],
        "pause": [13.1489458, 10.5460385, 11.4640105],
        "prolungamenti": [8.61482657, 63.7044968, 4.13456117],
        "pause_piene": [8.84153253, 14.8286938, 2.06728059],
        "segnali_discorsivi": [43.0741328, 41.7023555, 51.3061455]
    }

    df = pd.DataFrame(data)

    # Lista ordinata dei fenomeni (variabili)
    metrics = ["ripetizioni", "reparandum", "pause", "prolungamenti", "pause_piene", "segnali_discorsivi"]

    # 2. Trasformazione del DataFrame in formato "Long"
    df_long = pd.melt(
        df, 
        id_vars=["corpus"], 
        value_vars=metrics, 
        var_name="Fenomeno", 
        value_name="Valore"
    )

    # Ordine categoriale richiesto per i corpora
    ordine_corpora = ["KIParla", "TrIttok_Gen", "TrIttok_PolSo"]
    df_long["corpus"] = pd.Categorical(df_long["corpus"], categories=ordine_corpora, ordered=True)
    df_long = df_long.sort_values(["Fenomeno", "corpus"]).reset_index(drop=True)

    # 3. Configurazione dello stile grafico
    sns.set_theme(style="whitegrid")
    
    # Creiamo una riga di subplots (1 riga, tanti pannelli quanti sono i fenomeni)
    # sharey=True garantisce che tutti i pannelli abbiano ESATTAMENTE la stessa scala verticale
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 7), sharey=True)

    # Tavolozza colori coerente
    colori = sns.color_palette("Set2", len(ordine_corpora))
    mappa_colori = dict(zip(ordine_corpora, colori))

    # 4. Ciclo su ogni pannello/variabile
    for idx, fenomeno in enumerate(metrics):
        ax = axes[idx]
        # Filtriamo i dati per il fenomeno corrente
        df_sub = df_long[df_long["Fenomeno"] == fenomeno]
        
        # Disegniamo i pallini. Usiamo l'asse X del pannello per distribuire i 3 corpora
        sns.scatterplot(
            data=df_sub,
            x="corpus",
            y="Valore",
            hue="corpus",
            palette=mappa_colori,
            s=160,               # Pallini grandi e leggibili
            legend=False,        # Nascondiamo la legenda globale poiché i nomi sono già sui punti
            ax=ax
        )
        
        # AGGIUNTA ETICHETTE TESTUALI ACCANTO AI PUNTI
        for _, row in df_sub.iterrows():
            # Coordinata X: nel plot categoriale corrisponde all'indice di posizione nell'ordine del plot
            x_pos = ordine_corpora.index(row["corpus"])
            y_pos = row["Valore"]
            
            # Scriviamo il nome del corpus leggermente spostato a destra rispetto al pallino (+0.15)
            ax.text(
                x_pos + 0.15,
                y_pos,
                row["corpus"],
                fontsize=9.5,
                fontweight="semibold",
                va="center",
                ha="left",
                alpha=0.9
            )

        # Personalizzazione estetica del singolo pannello
        ax.set_title(fenomeno.replace("_", " ").title(), fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("") # Rimuoviamo l'etichetta X interna che sarebbe ridondante
        
        # Pulizia dell'asse X inferiore (i nomi sono già sui punti, lasciamo solo degli indicatori puliti)
        ax.set_xticklabels([]) 
        ax.set_xlim(-0.4, 2.8) # Spazio orizzontale calcolato per non far tagliare le scritte di testo

    # Personalizzazione dell'asse Y globale (presente solo sul primo pannello a sinistra grazie a sharey)
    axes[0].set_ylabel("Valore / Tasso normalizzato (ogni 100 token)", fontsize=13, fontweight="bold", labelpad=12)
    
    # Titolo generale della figura
    fig.suptitle("Profilo delle Disfluenze nei Corpora (Variabili Separate su Scala Comune)", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()

    # 5. Salvataggio e visualizzazione
    output_filename = "dotplot_sub plots_separati.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Grafico a pannelli separati salvato in: {output_filename}")
    plt.show()

if __name__ == "__main__":
    genera_dotplot_separato()