from __future__ import annotations  # Postpone evaluation of type hints (store them as strings)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TitanicPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler() # used to standardize numerical features (fit on train, apply on val/test)
        self.feature_columns = None # stores final feature names after encoding to ensure consistent columns order
        self.stats = {} # dictionary to store computed training statistics


    def _engineer_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Extract passenger title (Mr, Mrs, Miss, etc.) from full name and standardize uncommon variants
        df["Title"] = df["Name"].str.extract(r',\s*([^.]+)\.')[0].str.strip()
        df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
        df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
        
        # Family variables engineering
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        # Fare handling (preventing negative values ​​before log)
        df["Fare"] = df["Fare"].clip(lower=0)
        
        # Convert Pclass from numeric to string so it is treated as a categorical feature in one-hot encoding
        df["Pclass"] = df["Pclass"].astype(str)
        
        return df

    # Calculating statistics on the training set only, gap filling and encoding
    def fit(self, df: pd.DataFrame):
        df = self._engineer_base(df)
        
        # Filling in missing values 
        # Fare - using median
        self.stats["fare_median"] = df["Fare"].median()
        df["Fare"] = df["Fare"].fillna(self.stats["fare_median"])
        
        # Embarked - using mode
        self.stats["embarked_mode"] = df["Embarked"].mode()[0] if not df["Embarked"].mode().empty else "S"
        df["Embarked"] = df["Embarked"].fillna(self.stats["embarked_mode"])
        
        # Age - using median of Title
        self.stats["title_age_medians"] = df.groupby("Title")["Age"].median().to_dict()
        self.stats["global_age_median"] = df["Age"].median() 
        df["Age"] = df.apply(
            lambda r: r["Age"] if pd.notna(r["Age"]) 
            else self.stats["title_age_medians"].get(r["Title"], self.stats["global_age_median"]), axis=1
        )
        
        # Feature Engineering
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        df["IsChild"] = (df["Age"] < 12).astype(int)
        
        # Apply log transformation to reduce skewness
        df["Fare"] = np.log1p(df["Fare"])
        df["FarePerPerson"] = np.log1p(df["FarePerPerson"])
        
        # Training the scalar on continuous variables
        num_cols = ["Age", "Fare", "FamilySize", "FarePerPerson"]
        self.scaler.fit(df[num_cols])
        
        # Removing irrelevant columns
        cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        features_df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # One-Hot Encoding
        self.feature_columns = pd.get_dummies(features_df).columns.tolist()
        return self
    
    # Applying the Train statistics on Validation set
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._engineer_base(df)
        
        # Filling in missing values ​​based on train statistics only
        df["Embarked"] = df["Embarked"].fillna(self.stats["embarked_mode"])
        df["Fare"] = df["Fare"].fillna(self.stats["fare_median"])
        df["Age"] = df.apply(
            lambda r: r["Age"] if pd.notna(r["Age"]) 
            else self.stats["title_age_medians"].get(r["Title"], self.stats["global_age_median"]), axis=1
        )
        
        # As done in train
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        df["IsChild"] = (df["Age"] < 12).astype(int)
        df["Fare"] = np.log1p(df["Fare"])
        df["FarePerPerson"] = np.log1p(df["FarePerPerson"])
        
        # Applying the trained scaler
        num_cols = ["Age", "Fare", "FamilySize", "FarePerPerson"]
        df[num_cols] = self.scaler.transform(df[num_cols])
        
        # Removing irrelevant columns and performing one-hot encoding
        cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        features_df = df.drop(columns=cols_to_drop, errors='ignore')
        encoded_df = pd.get_dummies(features_df)
        
        # Add missing feature columns (from training set) with default 0 to ensure consistent model input
        for col in self.feature_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        
        return encoded_df[self.feature_columns].values.astype(np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
    
    def get_target(self, df: pd.DataFrame) -> np.ndarray:
        return df["Survived"].values.astype(np.float32)