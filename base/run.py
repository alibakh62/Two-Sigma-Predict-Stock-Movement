import os
import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews

try:
    del market_train_df, news_train_df
except:
    try:
        env = twosigmanews.make_env()
    except:
        pass
finally:
    (market_train_df, news_train_df) = env.get_training_data()

def make_prediction(predictions_df, model, X):
    predictions_df.confidenceValue = model.predict(X)
    predictions_df.confidenceValue = predictions_df.confidenceValue.clip(-1,1)

def main():
    dataprep = DataPreparation()
    X, y = dataprep.get_Xy_train(market_train_df, news_train_df)
    model = Model("CustomModel", None)
    model.fit(X, y)
    del X, y
    days = env.get_prediction_days()
    cnt = 0
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        X_test = dataprep.get_X_test(market_obs_df, news_obs_df)
        make_prediction(predictions_template_df, model, X_test)
        del X_test
        env.predict(predictions_template_df)
        cnt += 1
        print("Prediction Day {}".format(cnt))
    print('Done!')
    env.write_submission_file()

if __name__ == "__main__":
    main()
