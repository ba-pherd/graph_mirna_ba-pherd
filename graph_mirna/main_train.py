import pandas as pd
import os
import zipfile

from graph_mirna.dataset import MiRNADataProcessor
from graph_mirna.RnaBERT import execute_rna_bert
from graph_mirna.Correlation import execute_corr_graph
from graph_mirna.P_value_matrix_neg import normalize_p_value
from graph_mirna.Graph_Conv import execute_graph_conv
from graph_mirna.Scalar_Product import  scalar_product
from graph_mirna.Train_RF import train_model_and_save

from prin_task_api_utils import TaskIOManager 


def execute_training():
    processor = MiRNADataProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )

    processor.load_data()
    processor.preprocess()
    processor.save_processed_data()

    execute_rna_bert()

    train_df = pd.read_csv(f"{processor.processed_data_path}/df_concat_final.csv", index_col=0)

    # Carica la matrice di correlazione filtrata e crea il grafo
    correlation_output_path = f"{processor.processed_data_path}/correlation_matrix_train.csv"
    execute_corr_graph(train_df, correlation_output_path)
    corr_matrix = pd.read_csv(correlation_output_path, index_col=0)

    # Normalizza i p-value e crea il grafo, salvando anche gli embedding complessivi
    graph_output_path = f"{processor.processed_data_path}/graph_edges_train_fold.csv"

    complexive_df = normalize_p_value(
        p_value_matrix=corr_matrix,
        output1=graph_output_path,
        path=processor.processed_data_path
    )


    edge_df = pd.read_csv(graph_output_path)

    graph_conv_output_path = f"{processor.processed_data_path}/graph_embeddings.csv"
    execute_graph_conv(edge_df, complexive_df, graph_conv_output_path)
    node_embeddings=pd.read_csv(graph_conv_output_path,index_col=0)

    prod_train_path=f"{processor.processed_data_path}/subj_embeddings_train.csv"

    scalar_product(train_df,node_embeddings , complexive_df,prod_train_path)

    prod_train=pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_train.csv" ,index_col=0)


    view_combinations = [
        #(['expr'], ['expr_test']),
        #(['expr', 'meta'],['expr_test', 'meta_test']),
        (['prod', 'meta'], ['prod_test', 'meta_test']),
        #(['expr', 'prod'], ['expr_test', 'prod_test']),
        #(['prod'], ['prod_test']),
        #(['expr', 'prod', 'meta'], ['expr_test', 'prod_test', 'meta_test']),
        #(['meta'], ['meta_test'])
    ]

    # Qui stai preparando i risultati aggregati per ogni combinazione di viste
    for selected_views_train, selected_views_test in view_combinations:
        accuracies = []  # Lista per le accuracies dei fold
        reports = []  # Lista per i report di classificazione dei fold

        train_model_and_save(
            df_train=train_df,
            df_prod_train=prod_train,
            selected_views_train=selected_views_train,
            fold_number="final",  # Adatta questo se stai usando fold separati
            model_output_path="data/models",
            save_path=processor.processed_data_path
        )

    TRINO_USER = os.getenv('TRINO_USER')
    TRINO_GROUP = os.getenv('TRINO_GROUP')
    TRINO_ENDPOINT = os.getenv('TRINO_ENDPOINT')
    TRINO_CATALOG = os.getenv('TRINO_CATALOG')
    TRINO_SCHEMA = os.getenv('TRINO_SCHEMA')
    TASK_SCOPE = os.getenv('TASK_SCOPE')
    TASK_APIS_BASE_URL = os.getenv('TASK_APIS_BASE_URL') or ""

    taskIOManager = TaskIOManager(task_api_base_url=TASK_APIS_BASE_URL)

    # Create a zip file with `the file saved in the previous steps
    with zipfile.ZipFile('model_files.zip', 'w') as zipf:
        zipf.write('data/models/rf_model_prod_meta_final.joblib')
        zipf.write('data/models/rf_features_prod_meta_final.joblib')
        zipf.write('data/processed/graph_embeddings.csv')


    with open('model_files.zip', 'rb') as model_file:
            taskIOManager.save_model(model_file=model_file, user_group=TRINO_GROUP)
