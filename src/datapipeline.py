import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler , OneHotEncoder , PolynomialFeatures, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression , Lasso, Ridge, SGDRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import RegressorMixin
from pprint import pprint

import pickle
from typing import List, Union
import logging 

class Modelpipeline:
    """
    A pipeline to take in a pandas dataframe and has some configurables 
    to tailor the model or how the data would be preprocessed
    """
    def __init__(
        self, 
        dataset:pd.DataFrame,
        dimension_reduction:str = None,
        clustering:str = None,
        additional_features_out:List[str] = None,
        test_size:float=0.3,
        random_state:int=42,
        features_out:List[str] = ['street_name', 'lease_commence_date','block','address','full_address']
        ) -> None:
        dataset_copy = dataset.copy(deep= True)
        self.dataset = dataset_copy
        self.features_out = features_out
        self.ADDITIONAL_DROP_FEATURES = additional_features_out
        self.dimension_reduction = dimension_reduction
        self.clustering = clustering
        self.test_size = test_size
        self.random_state = random_state

        params_to_check = ['model','dimension_reduction','clustering']

        self.logger = logging.getLogger(__name__)

        # Check valid
        # for item in params_to_check:
        #     if self._check_valid(item, self.{item}) == False:
        #         raise SyntaxError('Incorrect value')

        self.dimension_reduction_model = PCA(n_components=0.95)
        self.clustering_model = DBSCAN(eps=1.5)
        self.logger.critical(f"Inspecting Dataframe \n Shape : {self.dataset.shape} \
            \n First Row : {self.dataset.head(1)} \n Columns : {self.dataset.columns}")

    def _check_valid(self,type:str,value:str)->bool:
        """Checks if the parameter is valid

        Args:
            type (str): Accepts model, dimension_reduction or clustering
            value (str): the value of the param, i.e if its model, then xgb, lgbm, etc

        Returns:
            bool: If params are valid
        """
        valid_list = {
            'model':['xgb','lgbm','lasso','linear_regression', 'ridge', 'elasticnet'],
            'dimension_reduction':['pca'],
            'clustering':['dbscan']
        }
        return value in valid_list[type]
        
    def _add_year(self, dataset:pd.DataFrame) -> None:
        """
        Adds the year feature into the dataset
        """
        try:
            dataset['year'] = pd.to_datetime(dataset['month']).dt.year
        except Exception as e:
            self.logger.critical(f"Unable to create year column due to error msg {e}")

    def _add_month(self, dataset:pd.DataFrame) -> None:
        """
        Adds the month feature from the existing month column,
        overwriting it
        """
        try:
            dataset['month'] = pd.to_datetime(dataset['month']).dt.month
        except Exception as e:
            self.logger.critical(f"Unable to change month column due to error msg {e}")

    def _add_district(self, dataset:pd.DataFrame) -> None:
        """
        Adds the district feature using the full_address column by taking
        the first 2 digits of the postal code in the full address. Fills na with 0
        """
        try:
            dataset['district'] = dataset['full_address'].apply(lambda x:x[-6:-4])
            dataset['district'] = pd.to_numeric(dataset['district'], errors='coerce')
            dataset['district'].fillna(0, axis=0, inplace=True)
            dataset['district'] = dataset['district'].astype(int)
        except Exception as e:
            self.logger.critical(f"Unable to add district column due to error msg {e}")
    
    def _add_features(self, dataset:pd.DataFrame) -> None:
        """
        Adds all of the features
        """
        self._add_year(dataset)
        self._add_month(dataset)
        self._add_district(dataset)

    def _convert_features(self, dataset:pd.DataFrame) -> None:
        """
        convert features from the dataset and replaces some text due to consistency issue
        """
        dataset['remaining_lease'] = dataset['remaining_lease'].astype(str).str[:2].astype(int)
        dataset['floor_area_sqm'] = dataset['floor_area_sqm'].astype(float)
        dataset['resale_price'] = dataset['resale_price'].astype(float)
        dataset['flat_type'] = dataset['flat_type'].str.replace('MULTI-GENERATION','MULTI GENERATION')

    def _drop_features_pipeline(self, which_features:List[str])->None:
        """
        Drop the features from the pipeline category
        """

        for feature_category in self.FEATURES_LIST:
            relevant_features = [feature for feature in feature_category if feature in \
                which_features]
            for feature in relevant_features:
                feature_category.remove(feature)


    def _dimension_reduction(self)->pd.DataFrame:
        """
        Uses the selected dimension reduction technique and replaces the original features
        with them
        """
        pass

    def _clustering_feature(self)->None:
        """
        Adds clustering label as feature after reducing dimensions
        """
        # PCA only takes in numerical values
        dataset_numerical = self.dataset.select_dtypes(exclude = 'object').loc[:, self.dataset.select_dtypes(exclude = 'object').columns != self.TARGET]
        dataset_dim_reduced = self.dimension_reduction_model.fit_transform(dataset_numerical)

        self.logger.critical(list(zip(["pca " + str(idx) for idx in \
            range(1,len(self.dimension_reduction_model.explained_variance_ratio_))] ,
            self.dimension_reduction_model.explained_variance_ratio_)))

        self.clustering_model.fit(dataset_dim_reduced)

        labels = self.clustering_model.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        self.logger.critical(f"Estimated number of clusters: {n_clusters_}")
        self.logger.critical(f"Estimated number of noise points: {n_noise_}")

        self.dataset['clustering_label'] = labels

    def _clustering_predict(self, inputs) ->pd.DataFrame:
        """
        Uses Kmeans to classify the inputs with the trained
        DBSCAN algorithm
        """
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=50)
        knn.fit(self.clustering_model.components_,
        self.clustering_model.labels[self.clustering_model.core_sample_indicies_])
        knn_labels = knn.predict(inputs)
        self.logger.critical(f"knn_labels: {knn_labels}")
        inputs['clustering_label'] = knn_labels

        return inputs


    def _pre_process(self)->None:
        """
        Prepares the dataset by adding, dropping, transforming features before
        splitting and training
        """

        self._add_features(self.dataset)
        self._convert_features(self.dataset)
        
        self.TARGET = "resale_price"
        self.NUM_FEATURES = self.dataset.select_dtypes(include=['float','int']).columns.tolist()
        self.NUM_FEATURES.remove(self.TARGET)
        self.DROP_FEATURES = ['street_name','block','address','full_address']
        self.ORDINAL_FEATURES = ["storey_range", "flat_type"]
        self.CAT_FEATURES = self.dataset.select_dtypes(exclude=['float','int']).columns.tolist()
        self.CAT_FEATURES = [feature for feature in self.CAT_FEATURES if feature not in self.ORDINAL_FEATURES]
        self.NEW_FEATURES = ['district']

        if self.dimension_reduction:
            self.NEW_FEATURES.append('clustering_label')

        self.TOTAL_FEATURES = self.dataset.columns.tolist() + self.NEW_FEATURES
        self.FEATURES = self.NUM_FEATURES + [self.TARGET] + self.CAT_FEATURES + self.ORDINAL_FEATURES + self.NEW_FEATURES
        self.FEATURES_LIST = [self.NUM_FEATURES, self.CAT_FEATURES, self.ORDINAL_FEATURES, self.NEW_FEATURES]
        assert len(self.TOTAL_FEATURES) == len(self.FEATURES)

        if len(self.features_out) !=0:
            # drop from pipeline
            self._drop_features_pipeline(self.features_out)
            # drop from dataset
            self.dataset.drop(columns=self.features_out, inplace=True)
        
        if self.ADDITIONAL_DROP_FEATURES:
            # drop from pipeline
            self._drop_features_pipeline(self.ADDITIONAL_DROP_FEATURES)
            # drop from dataset
            self.dataset.drop(columns=self.ADDITIONAL_DROP_FEATURES, inplace=True)

        if self.dimension_reduction:
            self._dimension_reduction()
        
        if self.clustering:
            self._clustering_feature()
        

    def _transform(self, dataset) -> np.array:
        """
        transforms the dataset by using the datapipe
        """
        dataset_transformed = self.datapipe.transform(dataset)

        return dataset_transformed

    def _fit(self, dataset)-> None:
        """
        Fits the datapipe with the dataset
        """

        ordinal_ranking_flat_type = ['MULTI GENERATION','EXECUTIVE', '5 ROOM','4 ROOM', '3 ROOM','2 ROOM', '1 ROOM']
        ordinal_ranking_storey_range = [ '01 TO 03','01 TO 05','04 TO 06', '06 TO 10', '07 TO 09', 
        '10 TO 12','11 TO 15', '13 TO 15','16 TO 18','16 TO 20','19 TO 21','21 TO 25', '22 TO 24', 
        '25 TO 27', '26 TO 30', '28 TO 30','31 TO 33','31 TO 35','34 TO 36','37 TO 39', '36 TO 40', 
        '40 TO 42', '43 TO 45', '46 TO 48','49 TO 51']

        self.ordinal_pipe = OrdinalEncoder(categories=[ordinal_ranking_storey_range,ordinal_ranking_flat_type])
        self.cat_pipe = OneHotEncoder(drop='first', sparse=False)
        self.num_pipe = StandardScaler()
        self.simpleimputer = SimpleImputer(strategy="median")

        self.datapipe = ColumnTransformer([
            ('cat_pipe',self.cat_pipe , self.CAT_FEATURES),
            ('num_pipe',self.num_pipe, self.NUM_FEATURES),
            ('ordinal_pipe', self.ordinal_pipe, self.ORDINAL_FEATURES),
            ('imputer', self.simpleimputer, self.NEW_FEATURES)
            ], remainder='passthrough')

        self.datapipe.fit(dataset)
    

    def fit_transform(self, dataset)-> np.array:
        """
        Fits and transforms the dataset with the datapipe
        """
        self._fit(dataset)
        dataset_transformed = self._transform(dataset)

        return dataset_transformed

    def preprocess(self, training_size:float=1, split:bool = True)->None:
        """
        Main function to preprocess the dataset and splits the datset
        """

        ############################################################
        # preps the dataset, adding, dropping features , etc
        ############################################################
        self._pre_process()

        self.logger.critical(f"Inspect dataset: {self.dataset.head()} \n \
            Columns : {self.dataset.columns}")

        ##############################
        # Seperates X and y
        ##############################
        if training_size !=1:
            training_samples = int(len(self.dataset)*training_size)
            y = self.dataset.resale_price.sample(training_samples, random_state = self.random_state)
            X = self.dataset.loc[:, self.dataset.columns != 'resale_price'].sample(training_samples, random_state = self.random_state)


        else:
            y = self.dataset.resale_price
            X = self.dataset.loc[:, self.dataset.columns != 'resale_price'] 

        if split == False:
            self.X_train, X_test, self.y_train, y_test= train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_test, y_test,test_size=0.5, random_state=self.random_state)
        else:
            self.X_train = X
            self.y_train = y

        self.X_transformed = self.fit_transform(self.X_train)
        self.logger.info(f"Shape of X transformed training set: {self.X_transformed.shape}")

        

    def train_model(self, model:Union[str, RegressorMixin])->None:
        """
        Trains the model
        """

        available_models = {
            'lgbm':lgb.LGBMRegressor(random_state = self.random_state),
            'xgb': xgb.XGBRegressor(random_state=self.random_state),
            'lr': LinearRegression(),
            'lasso': Lasso(),
            'ridge' : Ridge(),
            'elasticnet': ElasticNet(),
            'randomforest': RandomForestRegressor(random_state=self.random_state),
            'decisiontree': DecisionTreeRegressor(random_state=self.random_state),
            'sgd' : SGDRegressor()
        }

        if isinstance(model, str):
            self.model = available_models[model]
        elif isinstance(model, RegressorMixin):
            self.model = model
        else:
            raise TypeError("Please either pass in a string or a Regression Estimator")

        self.model.fit(self.X_transformed, self.y_train)

    def report_metrics(self, cv :int = 10)->dict:
        """
        Returns metrics for the chosen model in a dictionary
        """
        train_rmse_list = []
        val_rmse_list = []
        cv = cv
        model_cv_score = cross_validate(self.model,self.X_transformed, self.y_train, scoring = 'neg_mean_squared_error', 
        cv=cv, n_jobs = -1,return_train_score= True )
        
        train_rmse_list.append(model_cv_score['train_score'])
        val_rmse_list.append(model_cv_score['test_score'])
        
        self.metrics = {
            'Model': self.model,
            'Mean Train RMSE': ((-model_cv_score['train_score'])**(1/2)).mean(),
            'Train MSE': model_cv_score['train_score'].tolist(),
            'Val MSE' : model_cv_score['test_score'].tolist(),
            'Mean Val RMSE': ((-model_cv_score['test_score'])**(1/2)).mean()
        }

        metrics_to_print = {
            'Model': self.model,
            'Mean Train RMSE': f"{((-model_cv_score['train_score'])**(1/2)).mean():,}",
            'Mean Val RMSE': f"{((-model_cv_score['test_score'])**(1/2)).mean():,}"
            # 'Train RMSE': [f"{(-number)**(1/2):,}" for number in model_cv_score['train_score'].tolist()],
            # 'Val RMSE' : [f"{(-number)**(1/2):,}" for number in model_cv_score['test_score'].tolist()]
        }

        pprint(metrics_to_print)
        self.plot_results(model_cv_score, str(self.model))

    def report_best_features(self, plot:bool = False)->Optional[dict]:
        """
        Returns a dictionary with the best features and its importance
        Using model's inbuilt feature or Lasso if unavailable 
        """
        new_cols = []

        for pipeline in self.datapipe.transformers_:
            cols = pipeline[2]
            cat = pipeline[1]

            if isinstance(cat, OneHotEncoder):
                cols = pipeline[1].get_feature_names_out().tolist()
            
            new_cols += cols

        # self.logger.critical(f"Feature importance: {self.model.feature_importances_.tolist()}")
        # self.logger.critical(f"New Cols: {new_cols}")

        feature_importance = self.model.feature_importances_.tolist()
        assert len(feature_importance) == len(new_cols)
        res = {'Features': new_cols, 'Importance': feature_importance}
        if plot == False:
            return res
        else:
            pd.DataFrame(res).sort_values(by='Importance', ascending=False).iloc[:10,:].plot(kind='bar',x = 'Features', \
                y='Importance', title= f"Feature Importance of {self.model}" )

    def tune(self, params:dict, cv:int = 3)-> None:
        """
        Perform hyperparameter tunning for the model
        """

        gridsearchcv = GridSearchCV(self.model,param_grid = params, scoring = 'neg_mean_squared_error', cv=cv)

        gridsearchcv.fit(self.X_transformed, self.y_train)
        self.model = gridsearchcv.best_estimator_
        self.logger.critical(f"Training done, best params: {gridsearchcv.best_params_}")

    def _pre_process_prediction(self, inputs:pd.DataFrame)-> pd.DataFrame:
        """
        Prepares the input by adding, dropping, transforming features before
        running inference
        """

        self._add_features(inputs)
        self._convert_features(inputs)


        if len(self.features_out) !=0:
            # drop from dataset
            inputs.drop(columns=self.features_out, inplace=True)
        
        if self.ADDITIONAL_DROP_FEATURES:
            # drop from dataset
            inputs.drop(columns=self.ADDITIONAL_DROP_FEATURES, inplace=True)

        if self.dimension_reduction:
            self._dimension_reduction()
        
        if self.clustering:
            inputs = self._clustering_predict(inputs)

        return inputs

    def _preprocess_prediction_raw(self, input:pd.DataFrame)->pd.DataFrame:
        """
        Preprocess prediction input from user with minimal information

        User will need to provide only the following data for inference:
        - Postal Code
        - Flat-type
        - Storey-range
        - Town

        in order to provide the following
        - nearest_mrt
        - flat_model
        - floor area sqm
        - year, month : fixed to current year and month
        - lat, long
        - full address

        
        """

        pass


    def predict(self, inputs:pd.DataFrame)-> Union[float,List[float]]:
        """
        Using the model to predict either batch or single data
        Returns either a list of predictions or a single prediction
        """
        self.logger.critical(f"Input: {inputs}")
        # adds , remove features
        processed_inputs = self._pre_process_prediction(inputs)
        
        try:
            inputs_transformed = self.datapipe.transform(processed_inputs)
            self.logger.critical(f"Input transformed: {inputs_transformed}")      
            pred = self.model.predict(inputs_transformed)

            return pred
        except Exception as e:
            self.logger.critical(f"Unable to generate prediction due to msg: {e}")



    def plot_results(self, score, title):
        
        train_score = score['train_score']
        test_score = score['test_score']
        
        train_score = (-train_score)**(1/2) 
        test_score = (-test_score)**(1/2)
        
        cv = len(score['score_time'])
        
        fig, ax = plt.subplots(figsize = (5,5))
        
        ax.plot(range(cv), train_score, 'r-', label = 'Training RMSE');
        ax.plot(range(cv), test_score, 'b--', label = 'Validation RMSE');
        ax.set_title(title)
        ax.set_xlabel('K-Folds')
        ax.set_ylabel('Error')
        ax.legend()
        plt.show()

    def save_model(self, title):
        """
        saved model in path
        """
        pickle.dump(self.model, open(f"{title}.pkl",'wb'))
    
    def load_model(self, model_path):
        """
        loads model from path
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
