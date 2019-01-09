import numpy as np
import pandas as pd

class DataPreparation:
    
    def __init__(self):
        self.market_drop_lst = ['time', 'assetName']
        self.news_drop_lst   = ['sourceTimestamp', 'firstCreated', 'sourceId', 'headline',
                                'subjects', 'audiences', 'time', 'assetName', 'takeSequence',
                                'assetCodes', 'headlineTag']
        self.news_cols_for_dummy = ["provider", "urgency", "marketCommentary"]
        self.DATE_FORMAT = "%Y-%m-%d"
        self.train_cols = None

    def _convert_date_to_string(self, data):
        data["dateIndex"] = data["time"].apply(lambda x: x.strftime(self.DATE_FORMAT))
        return data
    
    def _normalize_assetName(self, data):
        data["assetNameIndex"] = data["assetName"].apply(lambda x: x.lower().replace(" ", ""))
        return data

    def _drop_redundant_cols(self, data, drop_lst):
        if drop_lst:
            data = data.drop(drop_lst, axis=1)
        return data

    def _add_dow_dummy_train(self, joined_data):
        joined_data["DOW"] = joined_data.index.get_level_values("dateIndex")
        joined_data["DOW"] = pd.to_datetime(joined_data["DOW"])
        joined_data["weekNum"] = joined_data["DOW"].dt.week
        joined_data["DOW"] = joined_data["DOW"].dt.day_name()
        joined_data["weekNum"] = joined_data["weekNum"].astype(str)
        joined_data = pd.get_dummies(joined_data, columns=["DOW", "weekNum"])
        return joined_data
    
    def _add_news_dummy_train(self, news_data):
        news_data = pd.get_dummies(news_data, columns=self.news_cols_for_dummy, drop_first=True)
        return news_data
    
    def preprocess_market(self, market_data):
        market_data = self._convert_date_to_string(market_data)
        market_data = self._normalize_assetName(market_data)
        market_data = self._drop_redundant_cols(market_data, drop_lst=self.market_drop_lst)
        market_data.set_index(["dateIndex", "assetNameIndex"], inplace=True)
        return market_data
    
    def preprocess_news(self, news_data):
        news_data = self._convert_date_to_string(news_data)
        news_data = self._normalize_assetName(news_data)
        news_data = self._drop_redundant_cols(news_data, drop_lst=self.news_drop_lst)
        news_data = self._add_news_dummy_train(news_data)
        news_data = news_data.groupby(["dateIndex", "assetNameIndex"], as_index=False).mean()
        news_data.set_index(["dateIndex", "assetNameIndex"], inplace=True)
        return news_data
    
    def join_market_news(self, market_data, news_data):
        market_data = self.preprocess_market(market_data)
        news_data   = self.preprocess_news(news_data)
        joined_data = market_data.merge(news_data, left_index=True, right_index=True, how='left')
        joined_data = joined_data.reset_index() #TODO: keep assetCode and assetName in the index for now
        joined_data = joined_data.set_index(["dateIndex", "assetNameIndex","assetCode"])
        joined_data = joined_data.fillna(0, axis=1) #TODO: impute NaN with 0 for now
        joined_data = self._add_dow_dummy_train(joined_data)
        try:
            joined_data = joined_data.drop(["universe"], axis=1)
        except:
            pass
        return joined_data
    
    def _add_missing_dummy_columns(self, test_df, train_cols):
        missing_cols = set(train_cols) - set(test_df.columns)
        for col in missing_cols:
            test_df[col] = 0
        return test_df
    
    def fix_testdf_columns(self, test_df):  
        if self.train_cols is None:
            raise Exception("First need to run 'get_Xy_train' to populate train data columns list!")
        else:
            train_cols = self.train_cols
        test_df = self._add_missing_dummy_columns(test_df, train_cols)
        # make sure we have all the columns we need
        assert(set(train_cols) - set(test_df.columns) == set())
        # removing extra columns in test
        extra_cols = set(test_df.columns) - set(train_cols)
        if extra_cols:
            test_df = test_df[train_cols]
        return test_df
    
    def get_train_data(self, market_data, news_data):
        train_data = self.join_market_news(market_data, news_data)
        return train_data

    def get_test_data(self, market_data, news_data):
        test_data = self.join_market_news(market_data, news_data)
        test_data = self.fix_testdf_columns(test_data)
        return test_data
    
    def get_Xy_train(self, market_data, news_data):
        df = self.get_train_data(market_data, news_data)
        X = df.loc[:, df.columns != 'returnsOpenNextMktres10']
        y = df.loc[:, df.columns == 'returnsOpenNextMktres10'].clip(-1,1)
        self.train_cols = X.columns
        return X, y  
    
    def get_X_test(self, market_data, news_data):
        X = self.get_test_data(market_data, news_data)
        return X  