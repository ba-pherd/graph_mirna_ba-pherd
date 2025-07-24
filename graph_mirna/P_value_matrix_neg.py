import pandas as pd
import numpy as np

import torch
import pickle


def normalize_p_value(p_value_matrix, output1,file_suffix="default",path="../data/processed/"):
    def normalize_embeddings(embeddings_tensor):
        """
        Normalizza gli embedding tra 0 e 1.
        """
        min_vals = embeddings_tensor.min(dim=0, keepdim=True).values
        max_vals = embeddings_tensor.max(dim=0, keepdim=True).values
        return (embeddings_tensor - min_vals) / (max_vals - min_vals)

    """
    Normalizza i p-value e costruisce un grafo con gestione di correlazioni negative.
    """


    normalized_p_value = p_value_matrix

    # Salva la mappatura dei nodi
    mapping_nodes = {k: v for k, v in enumerate(p_value_matrix.columns)}
    with open("../data/processed/mapping_miRNA", 'wb') as file:
        pickle.dump(mapping_nodes, file)
    print("Contenuto di mapping_nodes:", mapping_nodes)

    # Carica gli embedding e normalizzali
    embeddings = torch.load('../data/processed/miRNA.pt',weights_only=False)
    embeddings = pd.DataFrame.from_dict(embeddings, orient='index')
    # Filtra gli embedding per includere solo i nodi presenti in p_value_matrix.columns
    embeddings = embeddings.loc[p_value_matrix.columns]
    embeddings_tensor = torch.tensor(embeddings.values, dtype=torch.float)

    print("p_value_matrix.columns:", p_value_matrix.columns.tolist())
    print("embeddings.index:", embeddings.index.tolist())




    embeddings_normalized = normalize_embeddings(embeddings_tensor)
    embeddings_normalized = pd.DataFrame(embeddings_normalized.numpy(), index=embeddings.index)



    # Creazione dataframe per grafo
    upper_tri = normalized_p_value.where(np.triu(np.ones(normalized_p_value.shape), k=1).astype(bool))
    stacked = upper_tri.stack()
    stacked = stacked[stacked != 0]

    # Gestione correlazioni
    sources = []
    targets = []
    weights = []
    artificial_nodes = {}
    new_node_counter = len(p_value_matrix)  # Indice per i nodi artificiali

    for (source, target), weight in stacked.items():
        abs_weight = abs(weight)

        if weight < 0:  # Se la correlazione è negativa, crea nodi artificiali
            non_source = f"non_{source}"
            non_target = f"non_{target}"

            # Aggiungi i nodi artificiali al dizionario
            if non_source not in artificial_nodes:
                # Accedi correttamente alla riga corrispondente a 'source' nel DataFrame degli embeddings
                artificial_nodes[non_source] = (1 - embeddings_normalized.loc[source].values).tolist()
            if non_target not in artificial_nodes:
                # Accedi correttamente alla riga corrispondente a 'target' nel DataFrame degli embeddings
                artificial_nodes[non_target] = (1 - embeddings_normalized.loc[target].values).tolist()

            # Collegamento tra nodo originale e nodo artificiale
            sources.extend([non_source, source])  # Usa il nome del nodo artificiale
            targets.extend([target, non_target])  # Usa il nome del nodo artificiale
            weights.extend([abs_weight, abs_weight])

        else:  # Se la correlazione è positiva, aggiungila normalmente
            sources.append(source)
            targets.append(target)
            weights.append(abs_weight)

    # Creazione dataframe per il grafo
    result = {
        'source': sources,
        'target': targets,
        'weight': weights
    }
    df_graph = pd.DataFrame(result)

    # Salva il grafo in un file CSV
    df_graph.to_csv(output1, index=False)

    # Creazione e salvataggio embedding dei nodi artificiali
    artificial_embeddings = pd.DataFrame.from_dict(artificial_nodes, orient='index')
    artificial_embeddings.to_csv(f"{path}artificial_embeddings_{file_suffix}.csv")
    complexive_embedding = pd.concat([embeddings_normalized, artificial_embeddings])
    complexive_embedding.to_csv(f"{path}complexive_embeddings_{file_suffix}.csv")
    print(artificial_embeddings.shape)
    return complexive_embedding

