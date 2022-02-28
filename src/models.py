import pandas as pd
import numpy as np
import random as ran
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class User:
    def __init__(self, user_id, importance_features, not_sincerity=False):
        self.user_id = user_id
        self.importance_features = importance_features
        self.survey = pd.DataFrame(
            columns=['no', 'temp', 'hum', 'lux', 'score'])
        self.not_sincerity = not_sincerity

    def save_survey(self, no, temp, hum, lux, score):
        self.survey = self.survey.append({
            "no": no,
            "temp": temp,
            "hum": hum,
            "lux": lux,
            "score": score
        }, ignore_index=True)

    def score(self, temp, hum, lux):
        _score = np.array([0, 0, 0])

        if len(self.importance_features) == 0:
            if self.not_sincerity:
                for idx in range(0, len(_score)):
                    _score[idx] = 20
            else:
                one_line = ran.randrange(0, 2)
                if one_line == 0:
                    s = ran.randrange(5, 20, 5)
                    for idx in range(0, len(_score)):
                        _score[idx] = s
                else:
                    for idx in range(0, len(_score)):
                        _score[idx] = ran.randrange(5, 20)
        else:
            if 'temp' in self.importance_features:
                if (temp >= 18) & (temp <= 20):
                    _score[0] = 20
                else:
                    err = 0
                    if temp <= 18:
                        err = 18 - temp
                    else:
                        err = temp - 20
                    _score[0] = 20 - round(err / 2)
            else:
                _score[0] = ran.randrange(15, 20)

            if 'hum' in self.importance_features:
                if (hum >= 40) & (hum <= 60):
                    _score[1] = 20
                else:
                    err = 0
                    if hum <= 40:
                        err = 40 - hum
                    else:
                        err = hum - 60
                    _score[1] = 20 - round(err / 5)
            else:
                _score[1] = ran.randrange(15, 20)

            if 'lux' in self.importance_features:
                if (lux >= 700) & (lux <= 1500):
                    _score[2] = 20
                else:
                    err = 0
                    if lux < 700:
                        err = 700 - lux
                    else:
                        err = lux - 1500
                    _score[2] = 20 - round(err / 100)
            else:
                _score[2] = ran.randrange(15, 20)

        return _score.sum()


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


class TargetRFModel(RandomForestRegressor):
    def __init__(self, n_estimators, max_features, oob_score, random_state, features, labels):
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            oob_score=oob_score, random_state=random_state
        )
        self.X_train, self.X_test, \
            self.y_train, self.y_test \
            = train_test_split(features, labels, test_size=0.2)

    def fit(self):
        super().fit(self.X_train, self.y_train.ravel())

    def predict(self, X=[]):
        if len(X) == 0:
            self.prediction = super().predict(self.X_test)
            self._mse = mean_squared_error(self.y_test, self.prediction)
        else:
            self.prediction = super().predict(X)

        return self.prediction

    @property
    def mse(self):
        return self._mse
