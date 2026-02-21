```mermaid
flowchart TD
    A[prepare_data(data_path, train_size, no_features, feature_config_path)] --> B{Data file exists?}
    B -- No + download enabled --> C[download_trading_data(...)]
    B -- No + download disabled --> D[Raise FileNotFoundError]
    B -- Yes --> E[load_trading_data(...)]
    C --> E
    E --> F[df.dropna()]
    F --> G[Split raw data: train_df_raw / test_df_raw]
    G --> H{no_features == true?}
    H -- Yes --> I[Return train_df_raw, test_df_raw]
    H -- No --> J{feature_config_path provided?}
    J -- Yes --> K[FeaturePipeline.from_yaml(...)]
    J -- No --> L[create_default_pipeline()]
    K --> M[pipeline.fit(train_df_raw)]
    L --> M
    M --> N[train_features = pipeline.transform(train_df_raw)]
    M --> O[test_features = pipeline.transform(test_df_raw)]
    N --> P[train_df = concat(train_df_raw, train_features)]
    O --> Q[test_df = concat(test_df_raw, test_features)]
    P --> R[Return train_df, test_df]
    Q --> R
```
