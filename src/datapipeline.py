from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ordinal_ranking_flat_type = ['MULTI GENERATION','EXECUTIVE', '5 ROOM','4 ROOM', '3 ROOM','2 ROOM', '1 ROOM']
ordinal_ranking_storey_range = [ '01 TO 03','01 TO 05','04 TO 06', '06 TO 10', '07 TO 09', '10 TO 12','11 TO 15', '13 TO 15',
        '16 TO 18','16 TO 20','19 TO 21','21 TO 25', '22 TO 24', '25 TO 27', '26 TO 30', '28 TO 30',
       '31 TO 33','31 TO 35','34 TO 36',  '37 TO 39', '36 TO 40', '40 TO 42', '43 TO 45', '46 TO 48','49 TO 51']
       

class DataPipeline():

    def __init__(self):
        ORDINAL_FEATURES = ["storey_range","flat_type"]
        CAT_FEATURES = ["town"]
        NUM_FEATURES = ["remaining_lease","floor_area_sqm","month"]
        self.TARGET = "resale_price"

        ordinal_pipe = OrdinalEncoder(categories=[ordinal_ranking_storey_range,ordinal_ranking_flat_type])
        cat_pipe = OneHotEncoder(drop='first', sparse=False)
        num_pipe = StandardScaler()

        self.datapipe = ColumnTransformer([
            ('cat_pipe',cat_pipe , CAT_FEATURES),
            ('num_pipe',num_pipe, NUM_FEATURES),
            ('ordinal_pipe', ordinal_pipe, ORDINAL_FEATURES)
            ])    

    def __dropcols(self, df):
        df.drop(columns=['_id','lease_commence_date','street_name','block', 'flat_model'], inplace=True)
        return df

    def __conversions(self, df):
        df['remaining_lease'] = df['remaining_lease'].astype(str).str[:2].astype(int)
        df['floor_area_sqm'] = df['floor_area_sqm'].astype(float)
        df['resale_price'] = df['resale_price'].astype(float)
        df['month'] = pd.to_datetime(df['month']).dt.year

        return df

    def __preprocess(self,df):
        df['flat_type'] = df['flat_type'].str.replace('MULTI-GENERATION','MULTI GENERATION')
        return df

    def __split(self, df:pd.DataFrame):
        y = df.pop(self.TARGET)
        X = df


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)

        return X_train, X_test, y_train, y_test

    def __run_ct(self, X_train, X_test):
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
        df = self.__dropcols(df)
        df = self.__conversions(df)
        df = self.__preprocess(df)
        X_train, X_test, y_train, y_test = self.__split(df)

        X_train, X_test = self.__run_ct(X_train, X_test)

        return X_train, X_test, y_train, y_test