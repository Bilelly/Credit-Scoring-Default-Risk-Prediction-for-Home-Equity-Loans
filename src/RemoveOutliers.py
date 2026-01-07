
import pandas as pd
import numpy as np


def remove_outliers(df, features, threshold=1.5):
    """
    Cette fonction utilise l'IQR pour détecter et supprimer les outliers.
    Le seuil par défaut est de 1,5 pour l'IQR (modifiable).

    Args:
    - df: DataFrame initial
    - features: liste des colonnes à vérifier pour les outliers
    - threshold: valeur seuil de l'IQR pour la détection des outliers

    Returns:
    - df_cleaned: DataFrame après suppression des outliers
    """
    df_cleaned = df.copy()

    for col in features:
        # Calcul de l'IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # Définir les seuils inférieurs et supérieurs
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Conserver les valeurs qui ne sont pas des outliers
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    return df_cleaned