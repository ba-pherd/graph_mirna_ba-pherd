def train_model_and_save(
    df_train,
    df_prod_train,
    selected_views_train,
    fold_number="default",
    model_output_path="../data/models",
    save_path="../data/processed"
):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import os
    from joblib import dump

    # === Preprocessing: costruzione viste ===
    df_expr_train = df_train.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_train = df_train[['age', 'sex', 'apoe4']].copy()
    df_meta_train['sex'] = df_meta_train['sex'].map({'female': 0, 'male': 1})
    df_diagnosis_train = df_train[['disease']]

    views_train = {
        "expr": df_expr_train,
        "meta": df_meta_train,
        "prod": df_prod_train,
        "diagnosis": df_diagnosis_train
    }

    # Unione delle viste selezionate
    dataset = views_train[selected_views_train[0]]
    for view in selected_views_train[1:]:
        dataset = dataset.merge(views_train[view], left_index=True, right_index=True)
    dataset = dataset.merge(df_diagnosis_train, left_index=True, right_index=True)

    X_train = dataset.drop(columns=['disease'])
    y_train = dataset['disease']

    # === Training ===
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)

    # === Calcolo e salvataggio della feature importance ===
    importances = rfc.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })
    feature_importance_df['Normalized_Importance'] = (
        feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()
    )
    feature_importance_df = feature_importance_df.sort_values(by='Normalized_Importance', ascending=False)

    # === Preparazione nomi file univoci ===
    view_name = "_".join(selected_views_train)
    model_filename = f"rf_model_{view_name}_{fold_number}.joblib"
    features_filename = f"rf_features_{view_name}_{fold_number}.joblib"
    importance_filename = f"feature_importance_{view_name}_{fold_number}.csv"

    # === Creazione directory se non esiste ===
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # === Salvataggi ===
    dump(rfc, os.path.join(model_output_path, model_filename))
    dump(X_train.columns.tolist(), os.path.join(model_output_path, features_filename))
    feature_importance_df.to_csv(os.path.join(save_path, importance_filename), index=False)

    print(f"Model, features, and feature importance saved for views: {selected_views_train}, fold: {fold_number}")
