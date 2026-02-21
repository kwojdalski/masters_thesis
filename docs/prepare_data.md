```mermaid
flowchart TD
    A["prepare_data inputs"] --> B{"Data file exists?"}
    B -- "No and download enabled" --> C["download_trading_data"]
    B -- "No and download disabled" --> D["raise FileNotFoundError"]
    B -- "Yes" --> E["load_trading_data"]
    C --> E
    E --> F["drop NaNs"]
    F --> G["split raw data into train and test"]
    G --> H{"no_features is true?"}
    H -- "Yes" --> I["return raw train and raw test"]
    H -- "No" --> J{"feature config path provided?"}
    J -- "Yes" --> K["FeaturePipeline.from_yaml"]
    J -- "No" --> L["create_default_pipeline"]
    K --> M["pipeline.fit on train raw only"]
    L --> M
    M --> N["transform train raw"]
    M --> O["transform test raw"]
    N --> P["concat raw train with train features"]
    O --> Q["concat raw test with test features"]
    P --> R["return train_df and test_df"]
    Q --> R
```
