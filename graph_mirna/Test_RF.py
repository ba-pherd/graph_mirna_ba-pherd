import numpy as np
import json
import pandas as pd
import os
from sklearn.metrics import classification_report
from joblib import load


def load_model_and_evaluate(
    df_test,
    df_prod_test,
    selected_views_test,
    fold_number="default",
    model_output_path="../data/models",
    save_path="../data/processed",
    view_name=None  # <--- nuovo parametro
):

    # === Preprocessing ===
    df_expr_test = df_test.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_test = df_test[['age', 'sex', 'apoe4']].copy()
    df_meta_test['sex'] = df_meta_test['sex'].map({'female': 0, 'male': 1})
    df_diagnosis_test = df_test[['disease']]

    views_test = {
        "expr_test": df_expr_test,
        "meta_test": df_meta_test,
        "prod_test": df_prod_test,
        "diagnosis_test": df_diagnosis_test
    }

    # Unione viste test
    dataset_test = views_test[selected_views_test[0]]
    for view in selected_views_test[1:]:
        dataset_test = dataset_test.merge(views_test[view], left_index=True, right_index=True)
    dataset_test = dataset_test.merge(df_diagnosis_test, left_index=True, right_index=True)



    X_test = dataset_test.drop(columns=['disease'])
    y_test = dataset_test['disease']


    # === Nome vista combinata o fornita ===
    if view_name is None:
        view_name = "_".join(view.replace("_test", "") for view in selected_views_test)

    model_path = os.path.join(model_output_path, f"rf_model_{view_name}_{fold_number}.joblib")
    features_path = os.path.join(model_output_path, f"rf_features_{view_name}_{fold_number}.joblib")

    # === Caricamento modello e feature ===
    model = load(model_path)
    feature_names = load(features_path)


    X_test = X_test[feature_names]  # Allineamento colonne

    # === Predizione e valutazione ===
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = model.classes_
    results = []
    # Associa predizione e probabilità all'indice originale del DataFrame
    for idx, probabilities in zip(X_test.index, probs):
        max_position = np.argmax(probabilities)
        predicted_class = classes[max_position]
        predicted_probability = round(float(probabilities[max_position]), 4)
        results.append({
            "index": idx,
            "predicted_class": predicted_class,
            "probability": predicted_probability
        })
    results_json = json.dumps(results, indent=2)
    with open("results.json", "w") as f:
        f.write(results_json)

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    return report_dict,results_json



def load_model_and_predict(
    df_test,
    df_prod_test,
    selected_views_test,
    fold_number="default",
    model_output_path="../data/models",
    save_path="../data/processed",
    view_name=None  # <--- nuovo parametro
):
    # === Preprocessing ===
    df_expr_test = df_test.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_test = df_test[['age', 'sex', 'apoe4']].copy()
    df_meta_test['sex'] = df_meta_test['sex'].map({'female': 0, 'male': 1})

    views_test = {
        "expr_test": df_expr_test,
        "meta_test": df_meta_test,
        "prod_test": df_prod_test,
    }

    # Unione viste test
    dataset_test = views_test[selected_views_test[0]]
    for view in selected_views_test[1:]:
        dataset_test = dataset_test.merge(views_test[view], left_index=True, right_index=True)

    X_test = dataset_test

    # === Nome vista combinata o fornita ===
    if view_name is None:
        view_name = "_".join(view.replace("_test", "") for view in selected_views_test)

    model_path = os.path.join(model_output_path, f"rf_model_{view_name}_{fold_number}.joblib")
    features_path = os.path.join(model_output_path, f"rf_features_{view_name}_{fold_number}.joblib")

    # === Caricamento modello e feature ===
    model = load(model_path)
    feature_names = load(features_path)

    X_test = X_test[feature_names]  # Allineamento colonne

    # === Predizione e valutazione ===
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = model.classes_
    results = []
    # Associa predizione e probabilità all'indice originale del DataFrame
    for idx, probabilities in zip(X_test.index, probs):
        max_position = np.argmax(probabilities)
        predicted_class = classes[max_position]
        predicted_probability = round(float(probabilities[max_position]), 4)
        results.append({
            "index": idx,
            "predicted_class": predicted_class,
            "probability": predicted_probability
        })
    results_json = json.dumps(results, indent=2)
    with open("results.json", "w") as f:
        f.write(results_json)

    return results_json