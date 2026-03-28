import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Any, Optional

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Створює нові ознаки на основі існуючих даних."""
    df = df.copy()
    
    # Категорії лояльності
    def get_loyalty(row):
        if row['Tenure'] >= 5 or row['NumOfProducts'] > 3:
            return 'Most_loyal'
        elif row['Tenure'] >= 2 or row['NumOfProducts'] > 1:
            return 'Few_years_aquainted'
        return 'New_clients'

    # Вікові категорії
    def get_age_cat(age):
        if age < 25: return '18-25'
        if age < 50: return '25-55'
        return 'older then 55'

    df['Clients loyality'] = df.apply(lambda r: get_loyalty(r), axis=1)
    df['AgeGroup'] = df['Age'].apply(get_age_cat)
    df['Age_IsActive'] = df['Age'] * df['IsActiveMember']
    df['Age_Balance'] = df['Age'] * df['Balance']
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Gender_Age'] = df['Gender'] * df['Age']
    
    return df

def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Масштабує числові ознаки за допомогою StandardScaler."""
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_categorical(train_df: pd.DataFrame, val_df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Кодує категоріальні ознаки за допомогою OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[cat_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    
    for df in [train_df, val_df]:
        df[encoded_cols] = encoder.transform(df[cat_cols])
        df.drop(columns=cat_cols, inplace=True)
        
    return train_df, val_df, encoder

def preprocess_data(target_col: Any, drop_cols: List, cat_cols: List, raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, Any]:
    """Основна функція, що координує весь процес препроцесингу для навчання."""
    # Створюємо ознаки
    df = create_features(raw_df)
    
    # Визначаємо колонки
    target_col = target_col
    drop_cols = [target_col] + drop_cols
    input_cols = [col for col in df.columns if col not in drop_cols]
    
    # Розбиття
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )
    
    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    
    # Кодування
    cat_cols = cat_cols
    train_inputs, val_inputs, encoder = encode_categorical(train_inputs, val_inputs, cat_cols)
    
    # Масштабування (опціонально)
    scaler = None
    if scaler_numeric:
        num_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
        train_inputs, val_inputs, scaler = scale_features(train_inputs, val_inputs, num_cols)
        
    return {
        'X_train': train_inputs,
        'train_targets': train_targets,
        'X_val': val_inputs,
        'val_targets': val_targets,
        'scaler': scaler,
        'encoder': encoder,
        'input_cols': train_inputs.columns.tolist()
    }
    
def preprocess_new_data(drop_cols: List, cat_cols: List, test_df: pd.DataFrame, scaler: Optional[StandardScaler], encoder: OneHotEncoder) -> pd.DataFrame:
    """Обробляє нові дані (тестові) за допомогою вже навчених скейлера та енкодера."""
    df = create_features(test_df)
    
    # Видаляємо зайве
    drop_cols = drop_cols
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Кодування
    cat_cols = cat_cols
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    df[encoded_cols] = encoder.transform(df[cat_cols])
    df.drop(columns=cat_cols, inplace=True)
    
    # Масштабування
    if scaler is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[num_cols] = scaler.transform(df[num_cols])
        
    return df