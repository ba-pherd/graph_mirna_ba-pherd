import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def execute_corr_graph(df, output_file):
    # Elimina le colonne non necessarie
    df_expr = df.drop(columns=['disease', 'age', 'sex', 'apoe4', 'country'], errors='ignore')

    # Ottieni le colonne dei miRNA
    miRNA_columns = df_expr.columns

    # Inizializza dizionari per salvare correlazioni e p-values
    corr_dict = {}
    p_value_dict = {}

    # Creazione della matrice di correlazione
    corr_matrix = pd.DataFrame(np.eye(len(miRNA_columns)), columns=miRNA_columns, index=miRNA_columns)
    p_value_matrix = pd.DataFrame(np.zeros((len(miRNA_columns), len(miRNA_columns))), columns=miRNA_columns, index=miRNA_columns)

    # Calcolo della correlazione solo per la matrice triangolare superiore
    for i in range(len(miRNA_columns)):
        for j in range(i + 1, len(miRNA_columns)):
            col1 = miRNA_columns[i]
            col2 = miRNA_columns[j]
            col1_col2 = df_expr[[col1, col2]].dropna()

            if len(col1_col2) >= 2:
                corr, p_value = pearsonr(col1_col2[col1], col1_col2[col2])
            else:
                corr, p_value = np.nan, np.nan

            # Popola la matrice di correlazione e p-value
            corr_matrix.loc[col1, col2] = corr
            corr_matrix.loc[col2, col1] = corr  # La matrice è simmetrica

            p_value_matrix.loc[col1, col2] = p_value
            p_value_matrix.loc[col2, col1] = p_value  # La matrice è simmetrica

    # Filtriamo direttamente la matrice delle correlazioni basandoci sul p-value
    corr_matrix[p_value_matrix > 0.05] = 0

    # Salva la matrice delle correlazioni filtrata
    corr_matrix.to_csv(output_file)

    print(f"✅ Matrice di correlazione salvata in {output_file}")

