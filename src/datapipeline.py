from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class AddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, year:bool = True, month:bool = True,postal_code:bool = True) -> None:
        self.year = year
        self.month = month
        self.postal_code = postal_code
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.year:
            X['year'] = pd.to_datetime(X['month']).dt.year
        if self.month:
            X['month'] = pd.to_datetime(X['month']).dt.month
        if self.postal_code:
            X['postal_code'] = X['full_address'].apply(lambda x:x[-6:])
            X['postal_code'] = pd.to_numeric(X['postal_code'], errors='coerce')
            X['postal_code'].fillna(0, axis=0, inplace=True)
            X['postal_code'] = X['postal_code'].astype(int)
        return X

class Conversion(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['remaining_lease'] = X['remaining_lease'].astype(str).str[:2].astype(int)
        X['floor_area_sqm'] = X['floor_area_sqm'].astype(float)
        return X

class DataPipeline():

    def __init__(self, addfeature:bool = True,seed:int=42,test_size:float=0.33):
        self.test_size = test_size
        self.seed = seed
        self.ORDINAL_FEATURES = ["storey_range"]
        self.CAT_FEATURES = ["town", "nearest_mrt", "flat_model","flat_type"]
        self.NUM_FEATURES = ["remaining_lease","floor_area_sqm","month", "year",\
             "lat", "long", "postal_code","nearest_distance_to_mrt"]
        self.TARGET = "resale_price"
        self.POSSIBLE_NA_FEATURE = ["postal_code"]
        self.ordinal_ranking_storey_range = [ '01 TO 03','01 TO 05','04 TO 06', '06 TO 10', '07 TO 09', '10 TO 12','11 TO 15', '13 TO 15',
        '16 TO 18','16 TO 20','19 TO 21','21 TO 25', '22 TO 24', '25 TO 27', '26 TO 30', '28 TO 30',
       '31 TO 33','31 TO 35','34 TO 36',  '37 TO 39', '36 TO 40', '40 TO 42', '43 TO 45', '46 TO 48','49 TO 51']

        self.ordinal_pipe = OrdinalEncoder(categories=[self.ordinal_ranking_storey_range])
        self.cat_pipe = OneHotEncoder(drop='first', sparse=False)
        self.num_pipe = StandardScaler()
        self.convert_pipe = Conversion()
        self.addfeature_pipe = AddFeatures()
        self.simpleimputer = SimpleImputer(strategy="median")

        if addfeature==False:
            self.NUM_FEATURES.remove('postal_code')
            self.NUM_FEATURES.remove('year')

        column_pipe = ColumnTransformer([
            ('cat_pipe',self.cat_pipe , self.CAT_FEATURES),
            ('num_pipe',self.num_pipe, self.NUM_FEATURES),
            ('ordinal_pipe', self.ordinal_pipe, self.ORDINAL_FEATURES)
            ], remainder='drop')

        add_features_pipe = Pipeline([
            ('add_features', self.addfeature_pipe),
            ('conversion', self.convert_pipe)
        ])

        if addfeature:
            self.datapipe = Pipeline([
                ('add_features_conversion', add_features_pipe),
                ('column_transformer', column_pipe)
            ]) 

        else:
            self.datapipe = Pipeline([
                ('conversion', self.convert_pipe),
                ('column_transformer', column_pipe)
            ]) 

    def _split(self, df:pd.DataFrame):
        y = df.pop(self.TARGET)
        X = df


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, \
            random_state=self.seed)

        return X_train, X_test, y_train, y_test

    def _run_ct(self, X_train, X_test):
        X_train = self.datapipe.fit_transform(X_train)
        X_test = self.datapipe.transform(X_test)

        return X_train, X_test

    def transform(self, df, y= None):
        """preprocess, transform and then split the dataset

        Parameters
        ----------
        X : pd.DataFrame
            the entire DataFrame
        y : None, Ignored
        """
        X_train, X_test, y_train, y_test = self._split(df)

        X_train, X_test = self._run_ct(X_train, X_test)

        return X_train, X_test, y_train, y_test
