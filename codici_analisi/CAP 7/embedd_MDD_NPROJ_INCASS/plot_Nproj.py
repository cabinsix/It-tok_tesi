import os
import pandas as pd
from conllu import parse_incr
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
ROOT_DIR = "conllu_files"
IGNORE_FOLDERS = {"markit", "Ittok", "gen", "polso", "postwita","kiparla","tr-it-tok"}
MAX_LENGTH_THRESHOLD = 50 

def process_corpora():
    for root, dirs, files in os.walk(ROOT_DIR):
        folder_name = os.path.basename(root)
        if folder_name in IGNORE_FOLDERS: continue
        for filename in files:
            if filename.endswith(".conllu"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for sentence in parse_incr(f):
                        if folder_name == "isdt":
                            if "isst_tanl" not in sentence.metadata.get("sent_id", ""):
                                continue 
                        yield sentence, folder_name

def get_subtree(node_id, children_map):
    subtree = {node_id}
    stack = [node_id]
    visited = {node_id}
    while stack:
        curr = stack.pop()
        if curr in children_map:
            for child in children_map[curr]:
                c_id = child['id']
                if c_id not in visited:
                    visited.add(c_id)
                    subtree.add(c_id)
                    stack.append(c_id)
    return subtree

def generate_binned_data():
    raw_data = []
    for sentence, corpus in process_corpora():
        all_tokens = [t for t in sentence if isinstance(t['id'], int)]
        words_only = [t for t in all_tokens if t['deprel'] != 'punct']
        sent_len = len(words_only)

        if sent_len == 0 or sent_len > MAX_LENGTH_THRESHOLD:
            continue

        if sent_len <= 15:
            bin_name = '1. Short (1-15)'
        elif sent_len <= 35:
            bin_name = '2. Medium (16-35)'
        else:
            bin_name = '3. Long (36-50)'

        children_map = {}
        id_to_token = {}
        for token in all_tokens:
            t_id = token['id']
            id_to_token[t_id] = token
            if token['head'] != 0:
                children_map.setdefault(token['head'], []).append(token)

        is_nonproj = 0
        for token_j in words_only:
            j = token_j['id']
            i = token_j['head']
            if i == 0: continue
            
            subtree_i = get_subtree(i, children_map)
            start, end = sorted((i, j))
            
            for v_id in range(start + 1, end):
                if v_id in id_to_token and id_to_token[v_id]['deprel'] != 'punct':
                    if v_id not in subtree_i:
                        is_nonproj = 1
                        break
            if is_nonproj: break

        raw_data.append({'Corpus': corpus, 'Bin': bin_name, 'Is_NonProj': is_nonproj})

    df_raw = pd.DataFrame(raw_data)
    df_stats = df_raw.groupby(['Corpus', 'Bin']).agg(
        N_Frasi=('Is_NonProj', 'count'),
        N_NonProj=('Is_NonProj', 'sum')
    ).reset_index()
    
    df_stats['% Non-Proj'] = (df_stats['N_NonProj'] / df_stats['N_Frasi']) * 100
    return df_stats

def plot_connected_bubble_chart(df):
    plt.figure(figsize=(11, 6))
    
    # --- ORDINAMENTO DINAMICO BASATO SULLE FRASI LUNGHE ---
    # Isoliamo la fascia Long per decidere l'ordine dell'asse X
    df_long = df[df['Bin'] == '3. Long (36-50)'].copy()
    # Ordina i corpora dal valore più basso di non-proiettività a quello più alto
    df_long_sorted = df_long.sort_values(by='% Non-Proj')
    corpus_order = df_long_sorted['Corpus'].tolist()
    
    # Se un corpus non avesse frasi lunghe (caso limite), lo aggiungiamo in coda
    all_corpora = df['Corpus'].unique()
    for c in all_corpora:
        if c not in corpus_order:
            corpus_order.append(c)
            
    print(f"Nuovo ordine calcolato sull'asse X (basato sulle frasi lunghe): {corpus_order}")
    
    # Mappatura delle posizioni X
    corpus_to_x = {corpus: i for i, corpus in enumerate(corpus_order)}
    df['X'] = df['Corpus'].map(corpus_to_x)
    
    # Stili delle fasce
    bin_styles = {
        '1. Short (1-15)': {'color': '#1f77b4', 'label': 'Short (1-15)'},
        '2. Medium (16-35)': {'color': '#ff7f0e', 'label': 'Medium (16-35)'},
        '3. Long (36-50)': {'color': '#2ca02c', 'label': 'Long (36-50)'}
    }
    
    # Disegniamo prima i fili e poi le bolle per ogni fascia
    for bin_name, style in bin_styles.items():
        sub_df = df[df['Bin'] == bin_name].sort_values(by='X')
        
        if sub_df.empty:
            continue
            
        # 1. DISEGNA IL FILO (Linea di collegamento)
        plt.plot(
            sub_df['X'], 
            sub_df['% Non-Proj'], 
            color=style['color'], 
            linestyle='-', 
            linewidth=2, 
            alpha=0.5,
            zorder=1  # Sta sotto le bolle
        )
        
        # 2. DISEGNA LE BOLLE
        sizes = np.sqrt(sub_df['N_Frasi']) * 18  # Dimensione proporzionale al numero di frasi
        
        plt.scatter(
            sub_df['X'], 
            sub_df['% Non-Proj'], 
            s=sizes, 
            color=style['color'], 
            alpha=0.8, 
            edgecolors='black',
            label=style['label'],
            zorder=2  # Sta sopra il filo
        )
        
        # 3. ETICHETTE DEL NUMERO DI FRASI (n=...)
        for _, row in sub_df.iterrows():
            plt.text(
                row['X'], 
                row['% Non-Proj'] + 0.6, 
                f"n={int(row['N_Frasi'])}", 
                ha='center', va='bottom', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
            )

    # Personalizzazione assi e griglia
    plt.xticks(range(len(corpus_order)), corpus_order, fontsize=11)
    plt.ylabel("% di Frasi Non Proiettive (Havelka)", fontsize=12)
    plt.xlabel("Corpora (Ordinati per non-proiettività crescente nelle frasi lunghe)", fontsize=12)
    plt.title("Progressione della Non-Proiettività Sintattica per Fasce di Lunghezza\n(I fili collegano le stesse fasce; la dimensione indica il numero di frasi)", fontsize=12, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title="Fasce di Lunghezza", loc="upper left")
    plt.ylim(-3, max(df['% Non-Proj']) + 7)
    
    plt.tight_layout()
    plt.savefig("progressione_non_proiettivita_con_filo.png", dpi=300)
    print("Grafico salvato come 'progressione_non_proiettivita_con_filo.png'")
    plt.show()

if __name__ == "__main__":
    df_plot = generate_binned_data()
    plot_connected_bubble_chart(df_plot)