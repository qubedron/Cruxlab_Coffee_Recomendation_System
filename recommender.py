import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def filter_by_equipment(user_owned_equipment: list, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters recipes where required_equipment is a subset of owned_equipment.
    """
    user_set = set(user_owned_equipment)
    # Ensure all recipe requirements are met by the user's inventory
    mask = recipes_df['required_equipment'].apply(lambda req: set(req).issubset(user_set))
    return recipes_df[mask].copy()


def recommend_popular(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Rule-based baseline: Recommends most popular recipes filtered by equipment.
    """
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available_recipes = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    # Calculate global popularity based on interaction counts
    popularity = train_df['recipe_id'].value_counts().reset_index()
    popularity.columns = ['recipe_id', 'count']
    
    # Merge popularity scores and sort available recipes
    res = available_recipes.merge(popularity, on='recipe_id', how='left').fillna(0)
    res = res.sort_values('count', ascending=False).head(n)
    
    return list(zip(res['recipe_id'], res['count'].astype(float)))


def recommend_content(
    user_id: str, 
    users_df: pd.DataFrame, 
    recipes_df: pd.DataFrame, 
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Content-based fallback: Uses Euclidean distance between flavor profiles for cold start.
    """
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available_recipes = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    taste_features = ['bitterness', 'sweetness', 'acidity', 'body']
    
    # Calculate Euclidean distance for flavor matching
    diffs = []
    for t in taste_features:
        diff = (available_recipes[f'taste_{t}'] - user_info[f'taste_pref_{t}'])**2
        diffs.append(diff)
    
    available_recipes['distance'] = np.sqrt(sum(diffs))
    # Invert distance to create a score where higher is better
    available_recipes['score'] = 1 / (1 + available_recipes['distance'])
    
    res = available_recipes.sort_values('score', ascending=False).head(n)
    return list(zip(res['recipe_id'], res['score']))


def recommend(
    user_id: str,
    users_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Hybrid Recommender: Uses LightGBM for warm users and content-based for cold start.
    """
    # 1. Mandatory hard filter by equipment
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    available_recipes = filter_by_equipment(user_info['owned_equipment'], recipes_df)
    
    # 2. Check for warm vs cold start
    user_history = train_df[train_df['user_id'] == user_id]
    model_path = 'models/coffee_model.pkl'
    
    if user_history.empty:
        # Strategy for new users: Content-based flavor matching
        return recommend_content(user_id, users_df, recipes_df, n)
    
    # 3. Use ML ranking for warm users if model is available
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X_predict = available_recipes.copy()
        
        # Prepare feature vector (must match training feature order)
        for t in ['bitterness', 'sweetness', 'acidity', 'body']:
            X_predict[f'taste_pref_{t}'] = user_info[f'taste_pref_{t}']
            X_predict[f'delta_{t}'] = abs(X_predict[f'taste_{t}'] - user_info[f'taste_pref_{t}'])
        
        X_predict['pref_strength_norm'] = user_info['pref_strength_norm']
        X_predict['strength_match'] = (X_predict['strength'] == user_info['preferred_strength']).astype(int)
        
        feature_cols = [
            'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
            'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 
            'pref_strength_norm', 'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 
            'strength_match'
        ]
        
        # Rank available candidates using LightGBM
        available_recipes['score'] = model.predict(X_predict[feature_cols])
        res = available_recipes.sort_values('score', ascending=False).head(n)
        
        return list(zip(res['recipe_id'], res['score']))
    
    # Global fallback if ML model is missing
    return recommend_popular(user_id, users_df, recipes_df, train_df, n)