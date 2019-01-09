import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class CustomModel:
    '''
    X_cols: Xtrain columns, to ensure columns order for \
    train & test are aligned before for scaling
    '''

    def __init__(self, droplst=[]):
        self.do_scale = True
        self.scalerX = None
        self.X_cols = None 
        self.kwargs = {'n_estimators': 100,
                       'verbose': 1,
                       'learning_rate': 0.1,
                       'max_depth': 3}
        self.droplst = droplst
        self._init_model()

    def _scaleX(self, X):
        # X: numpy ndArray
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalerX = scaler
        return X_scaled

    def _prepare_data(self, X, y):
        X_ = X.copy()
        if self.droplst:
            X_ = X_.drop(self.droplst, axis=1)
        self.X_cols = X_.columns
        if self.do_scale:
            X_ = self._scaleX(X_.values)
        return X_, y

    def _predict(self, X):
        X_ = X[self.X_cols].copy()
        if self.do_scale:
            X_ = self.scalerX.transform(X_.values)
        ypred = self.model_.predict(X_)
        return ypred

    def _init_model(self):
        if self.kwargs is None:
            self.model_ = GradientBoostingRegressor()
        else:
            self.model_ = GradientBoostingRegressor(**self.kwargs)
    
    def fit(self, X, y):
        X, y = self._prepare_data(X, y)
        self.model_.fit(X, y)

    def predict(self, X):
        ypred = self._predict(X)
        return ypred