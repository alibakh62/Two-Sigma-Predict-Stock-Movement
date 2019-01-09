from custommodel import CustomModel
from sklearn.utils.validation import column_or_1d
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class Model:
    
    def __init__(self, model_name, kwargs):
        model_dict = {"LinearRegression": LinearRegression,
                      "ElasticNetCV": ElasticNetCV,
                      "GradientBoostingRegressor": GradientBoostingRegressor,
                      "RandomForestRegressor": RandomForestRegressor,
                      "CustomModel": CustomModel}
        self.model_name = model_name
        if kwargs is None:
            self.model = model_dict[model_name]()
        else:
            self.model = model_dict[model_name](**kwargs)
    def fit(self, X, y):
        y = column_or_1d(y)
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)