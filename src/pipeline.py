import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

# 1) Custom Transformers
class TitleExtractor(BaseEstimator, TransformerMixin):
    rare_titles = {'Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'}
    title_map   = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        titles = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        titles = titles.replace(self.rare_titles, 'Rare')
        titles = titles.replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
        X['Title'] = titles.map(self.title_map).fillna(0).astype(int)
        return X.drop(columns=['Name'])

class FamilySizeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['FamilySize'] = X['SibSp'].fillna(0) + X['Parch'].fillna(0) + 1
        X['IsAlone']    = (X['FamilySize'] == 1).astype(int)
        return X.drop(columns=['SibSp','Parch'])

class DeckImputer(BaseEstimator, TransformerMixin):
    deck_order = {'DE':1,'B':2,'C':3,'A':4,'G':5,'F':6}

    def fit(self, X, y=None):
        df = X.copy()
        df['Deck0'] = df['Cabin'].str[0].fillna('Unknown')
        modes = (
            df[df['Deck0']!='Unknown']
              .groupby('Pclass')['Deck0']
              .agg(lambda s: s.mode().iat[0])
        )
        self.deck_mode_ = modes.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X['Deck0'] = X['Cabin'].str[0].fillna('Unknown')
        X['Deck0'] = X.apply(
            lambda r: self.deck_mode_.get(r['Pclass'], 'Unknown')
                      if r['Deck0']=='Unknown' else r['Deck0'],
            axis=1
        )
        X = X[X['Deck0']!='T']
        X['Deck0'] = X['Deck0'].replace({'D':'DE','E':'DE'})
        X['DeckOrdinal'] = X['Deck0'].map(self.deck_order).fillna(0).astype(int)
        return X.drop(columns=['Cabin','Deck0'])

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_ = X.copy()
        self.age_median_ = X_['Age'].median()
        self.fare_median_ = X_['Fare'].median()
        fare_filled = X_['Fare'].fillna(self.fare_median_)
        _, self.fare_bins_ = pd.qcut(fare_filled, 4, retbins=True, duplicates='drop')
        return self

    def transform(self, X):
        X = X.copy()
        # Impute
        X['Age'] = X['Age'].fillna(self.age_median_)
        X['Fare'] = X['Fare'].fillna(self.fare_median_)
        # Features
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']
        X['FareBin'] = pd.cut(
            X['Fare'],
            bins=self.fare_bins_,
            labels=False,
            include_lowest=True
        ).astype(int)
        X['AgeBin'] = pd.cut(
            X['Age'],
            bins=[0,12,18,35,60,80],
            labels=False,
            include_lowest=True
        ).astype(int)
        return X

# 2) ColumnTransformer Setup
agg_cols = ['FamilySize','IsAlone','FarePerPerson','AgeBin','FareBin','DeckOrdinal']

sex_encoder = OrdinalEncoder(categories=[['male','female']], dtype=int)
embarked_encoder = OrdinalEncoder(categories=[['S','C','Q']], dtype=int)
title_encoder = OrdinalEncoder(categories=[[[1,2,3,4,5]]], dtype=int)

preprocessor = ColumnTransformer(transformers=[
    ('sex_ord',     sex_encoder,      ['Sex']),
    ('embarked_ord',embarked_encoder, ['Embarked']),
    ('title_ord',   title_encoder,    ['Title']),
    ('pass_agg',    'passthrough',    agg_cols),
], remainder='drop')

# 3) Full Pipeline

titanic_pipeline = Pipeline([
    ('title',    TitleExtractor()),
    ('family',   FamilySizeAdder()),
    ('deck',     DeckImputer()),
    ('features', FeatureEngineer()),
    ('preproc',  preprocessor),
    ('clf',      LogisticRegression(max_iter=1000, random_state=42)),
])
