import numpy as np
import pandas as pd
import random as ran
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class RFSimulation():
    def __init__(self, users, features):
        self.models = np.array([])
        self.users = users
        self.features = features
        self._record = pd.DataFrame(columns=['user id',
                                             'imp features',
                                             'RF imp features',
                                             'mse'])

    def run(self):
        for user in self.users:
            df = user.survey
            X = df[self.features].to_numpy().copy()
            y = df[['score']].to_numpy().copy()
            RF_imp_features = None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)

            RFModel = RandomForestRegressor(n_estimators=100,
                                            max_features=3,
                                            oob_score=False, random_state=531)
            RFModel.fit(X_train, y_train.ravel())
            self.models = np.append(self.models, RFModel)
            prediction = RFModel.predict(X_test)

            mse = mean_squared_error(y_test, prediction)

            feature_importance = RFModel.feature_importances_
            if len(set(feature_importance)) == 1:
                RF_imp_features = []
            else:
                feature_importance = feature_importance / feature_importance.max()

            sorted_idx = np.argsort(feature_importance)

            _features = self.features[sorted_idx][::-1]
            user_id = user.user_id
            user_imp_features = user.importance_features

            imp_length = len(user_imp_features)

            if RF_imp_features == None:
                if imp_length == 0:
                    RF_imp_features = [_features[0]]
                else:
                    RF_imp_features = _features[:imp_length]

            _imp_features = RF_imp_features
            imp_features = list()

            for _ in ['no', 'temp', 'hum', 'lux']:
                if _ in _imp_features:
                    imp_features.append(_)

            self._record = self._record.append({
                "user id": user_id,
                "mse": mse,
                "imp features": ",".join(user_imp_features),
                "RF imp features": ",".join(imp_features)
            }, ignore_index=True)


def generate_features():
    features = ['temp', 'hum', 'lux']
    imp_features = list()
    for f in features:
        is_in = ran.randrange(0, 2)
        if is_in == 0:
            imp_features.append(f)

    if len(imp_features) == 0:
        is_in = ran.randrange(0, 3)
        imp_features.append(features[is_in])

    return imp_features
