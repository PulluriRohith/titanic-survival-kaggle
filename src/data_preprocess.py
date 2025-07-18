import pandas as pd
import numpy as np

DROP_COLS = [
    'PassengerId', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch', 'HasCabin', 'Deck'
]

gender_mapping   = {'male': 0, 'female': 1}
title_mapping    = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
rare_titles      = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
deck_ord_mapping = {'DE': 1, 'B': 2, 'C': 3, 'A': 4, 'G': 5, 'F': 6}

def preprocess_titanic_survival(df: pd.DataFrame, stats: dict | None = None):
    df = df.copy()
    fitting = stats is None

    # 1) Title extraction and mapping
    df['Title'] = (
        df['Name']
          .str.extract(r' ([A-Za-z]+)\.', expand=False)
          .replace(rare_titles, 'Rare')
          .replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
          .map(title_mapping)
    )

    # 2) FamilySize & IsAlone
    df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # 3) Fare impute (if needed)
    if fitting:
        stats = {}
        stats['fare_median'] = df['Fare'].median()
        stats['fare_bins'] = pd.qcut(df['Fare'].fillna(stats['fare_median']), 4, retbins=True)[1]
    df['Fare'] = df['Fare'].fillna(stats['fare_median'])

    # 4) FareBin & FarePerPerson
    df['FareBin'] = pd.cut(
        df['Fare'],
        bins=stats['fare_bins'],
        labels=False,
        include_lowest=True
    ).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # 5) Embarked mapping
    df['Embarked'] = df['Embarked'].map(embarked_mapping).fillna(-1).astype(int)

    # 6) Deck extraction, imputation, and ordinal encoding
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')
    if fitting:
        stats['deck_mode'] = (
            df[df['Deck'] != 'Unknown']
              .groupby('Pclass')['Deck']
              .agg(lambda x: x.mode().iat[0])
              .to_dict()
        )
    df['Deck'] = df.apply(
        lambda r: stats['deck_mode'].get(r['Pclass'], 'Unknown')
                  if r['Deck'] == 'Unknown' else r['Deck'],
        axis=1
    )
    # Remove rows where Deck is 'T'
    df = df[df['Deck'] != 'T']
    # Merge decks 'D' and 'E' into 'DE'
    df['Deck'] = df['Deck'].replace({'D': 'DE', 'E': 'DE'})
    df['Deck_Ordinal'] = df['Deck'].map(deck_ord_mapping).fillna(0).astype(int)

    # 7) Sex mapping
    df['Sex'] = df['Sex'].map(gender_mapping).fillna(0).astype(int)

    # 8) Age impute & bin
    if fitting:
        stats['age_mean'] = df['Age'].mean()
    df['Age'] = df['Age'].fillna(stats['age_mean'])
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 80],
        labels=False,
        include_lowest=True
    ).astype(int)

    # 9) Drop unused columns
    df.drop(columns=[c for c in DROP_COLS if c in df], inplace=True)

    # 10) Zero-fill any leftover NaNs
    df = df.fillna(0)

    return df, stats