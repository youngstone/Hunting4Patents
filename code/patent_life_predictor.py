import pandas as pd
import numpy as np
import re
import cPickle as pkl
import json
from collections import Counter
from datetime import datetime, date
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, \
    recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re
from patent_tokenizer import PatentTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from pandas.core.reshape import get_dummies


def plot_roc(X, y, clf_class, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    fig = plt.figure(figsize=(12, 8))

    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)'
                 % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)'
             % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# below are some utility functions

def events_count(entry):
    code = []
    if type(entry) != float:
        for event in entry:
            code.append(event['Code'])
    return Counter(code)


def get_code(entry):
    code = []
    if type(entry) != float:
        for event in entry:
            code.append(event['Code'])
    return code


def get_date(entry):
    date = []
    if type(entry) != float:
        for event in entry:
            dt = event['Date']
            dt_dt = datetime.strptime(dt, '%b %d, %Y')
            date.append(dt_dt)
    return date


def determine_early_termination(fil_dat, code):
    today = date.today()
    screening_day = datetime(today.year - 20, today.month, today.day)
    if fil_dat.to_datetime() < screening_day:
        return 0
    elif 'FP' in code:
        return 1
    else:
        return -1


class PatentLongevityPredictor(object):

    def __init__(self):
        self.df = None
        self.df_transformed = None
        self.df_train = None
        self.df_test = None
        self.train_features = None
        self.train_target = None
        self.patent_status = None
        self.features = None
        self.target = None
        self.predict_model = None
        self.vectorizer = None
        self.test_prediction = None
        self.patent_tokenizer = None
        self.patent_vectors = None
        self.predicted_status = None

    def initialization(self, df_input):
        print "initializing PatentLongevityPredictor..."

        self.df = df_input

        cond = df_input['Filing Date'].apply(lambda x: type(x)) \
            == pd.tslib.NaTType
        self.df = df_input[~cond]

        df_transformed = self.transform(self.df)
        self.df_transformed = df_transformed

        # print df_transformed.head()

        self.df_train = df_transformed[df_transformed['Patent Status'] != -1]
        self.df_test = df_transformed[df_transformed['Patent Status'] == -1]

        self.train_features = self.df_train.drop(['Patent Status'], axis=1)
        self.train_target = self.df_train['Patent Status']

        self.test_features = self.df_test.drop(['Patent Status'], axis=1)

        best_estimator_ = self.build_model(self.train_features,
                                           self.train_target)

        self.predict_model = best_estimator_

        return

    def predict(self):
        print "predicting..."
        features = self.df_transformed.drop(['Patent Status'], axis=1)
        rf_best = self.predict_model
        y_predict = rf_best.predict(features)
        print "number of predictions:"
        print y_predict.shape

        self.prediction = y_predict
        print "predictions:"
        print y_predict

        print "=========="
        print type(self.patent_status.values), self.patent_status.values.shape
        print type(self.prediction), self.prediction.shape

        df_result = self.df

        a = []
        pagerank = self.df['PageRank'].values

        for fact, pred, pr in zip(self.patent_status.values,
                                  self.prediction, pagerank):
            if fact == 0:
                a.append('Expired Naturally')
            if fact == 1:
                a.append('Expired Early')
            if fact == -1:
                if pred == 1:
                    a.append('Predicted Yes')
                elif pred == 0:
                    a.append('Predicted No')

        predicted_events = pd.Series(data=a)

        df_result['Expiration Prediction'] = predicted_events
        df_result = df_result[['Patent Number', 'Expiration Prediction']]
        df_result.to_pickle('../data/patent_prediction.pkl')

        print df_result['Expiration Prediction'].value_counts()
        return

    def transform(self, df):
        print "transforming input dataframe..."

        # cond = df['Filing Date'].apply(lambda x: type(x)) == pd.tslib.NaTType
        # df = df[~cond]
        ###
        # print df['Filing Date'].values[-10:]
        # min_day = df['Filing Date'].min()
        # print min_day
        # print type(min_day)
        # min_day = min_day.to_datetime()
        # print min_day
        # print type(min_day)
        # print df.info()

        # df['Filing Date'] = df['Filing Date'].apply(lambda x: \
        #                     min_day if type(x) == pd.tslib.NaTType else x)
        ###
        print df['Filing Date'].values[-10:]

        cod = df['legal-events'].apply(get_code)
        dat = df['legal-events'].apply(get_date)
        fil_dat = df['Filing Date']

        end = []
        for d, e in zip(fil_dat, cod):
            end.append(determine_early_termination(d, e))

        self.patent_status = pd.Series(data=end)
        df['Patent Status'] = self.patent_status

        feature_names = [u'Patent Number',
                         u'Title',
                         u'US Patent References',
                         u'Abstract',
                         u'Primary Class',
                         u'Claims',
                         u'backward-citations',
                         u'npl-citations',
                         u'backward-citations-by-examiner',
                         u'backward-citations-by-inventor',
                         u'forward-citations-by-examiner',
                         u'forward-citations-by-inventor',
                         u'Patent Status']

        df = df[feature_names]

        df = df.rename(columns={'Patent Number': 'Number'})

        df['num_of_claims'] = df['Claims'].apply(len)

        # df['num_of_claims'] = df['Claims'].apply(lambda x: \
        #                             0 if type(x) == float else len(x))

        fwd_cit_ratio_inv_exm = []
        fwd_cit_inv = df['forward-citations-by-inventor']
        fwd_cit_exm = df['forward-citations-by-examiner']
        fwd_cit_ratio_inv_exm = [float(i) / (i + e)
                                 for i, e in zip(fwd_cit_inv, fwd_cit_exm)]

        df['fwd_cit_ratio_inv_exm'] = fwd_cit_ratio_inv_exm
        df['fwd_cit_ratio_inv_exm'] = \
            df['fwd_cit_ratio_inv_exm'].fillna(float(1))

        bwd_cit_ratio_inv_exm = []
        bwd_cit_inv = df['backward-citations-by-inventor']
        bwd_cit_exm = df['backward-citations-by-examiner']
        bwd_cit_ratio_inv_exm = [float(i) / (i + e)
                                 for i, e in zip(bwd_cit_inv, bwd_cit_exm)]

        df['bwd_cit_ratio_inv_exm'] = bwd_cit_ratio_inv_exm
        df['bwd_cit_ratio_inv_exm'] = \
            df['bwd_cit_ratio_inv_exm'].fillna(float(0))

        df['bwd_pat_cit_count'] = df['backward-citations'].apply(len)

        # df['bwd_pat_cit_count'] = df['backward-citations'].apply(lambda x: \
        #                             0 if type(x) == float else len(x))
        df['Primary Class'] = df['Primary Class'].apply(lambda x: x[0])
        # df['Primary Class'] = df['Primary Class'].apply(lambda x: \
        #                                 np.nan if type(x) == float else x[0])
        df['npl-citations'] = df['npl-citations'].fillna(0)

        df['Claims'] = df['Claims'].apply(lambda x: ' '.join(x))

        drop_columns = ['US Patent References',
                        'forward-citations-by-inventor',
                        'forward-citations-by-examiner',
                        'backward-citations-by-inventor',
                        'backward-citations-by-examiner',
                        'backward-citations']

        df_model = df.drop(drop_columns, axis=1)

        df_rf_features = self.add_claim_vectorization(df_model)

        # print "something new"
        # print df_rf_features.columns.tolist().index('Patent Status')

        return df_rf_features

    def add_claim_vectorization(self, df_model):
        print "adding claim vectorization..."

        df_model = df_model

        file_path = '../data/df_model.pkl'
        df_model.to_pickle(file_path)

        patent_tokenizer = PatentTokenizer()
        patent_tokenizer.set_df(file_path)
        patent_tokenizer.set_vectors()

        patent_claims_vectorizer = patent_tokenizer.get_claims_vectorizer()
        patent_claims_vectors = patent_tokenizer.get_claims_vectors()

        self.patent_tokenizer = patent_claims_vectorizer
        self.patent_vectors = patent_claims_vectors

        claims_features_name = patent_claims_vectorizer.get_feature_names()

        df_claims_features = pd.DataFrame(data=patent_claims_vectors,
                                          index=df_model.index,
                                          columns=claims_features_name)

        df_merge = pd.merge(df_model, df_claims_features,
                            how='inner', left_index=True,
                            right_index=True)

        non_needed = ['Number', 'Title', 'Abstract', 'Claims']
        df_temp = df_merge.drop(non_needed, axis=1)

        dummy_class = get_dummies(df_temp['Primary Class'],
                                  dummy_na=True,
                                  prefix='primary_class')

        df_rf_features = pd.merge(df_temp, dummy_class,
                                  how='inner', left_index=True,
                                  right_index=True)

        df_rf_features = df_rf_features.drop(['Primary Class'], axis=1)

        return df_rf_features

    def build_model(self, features, target):
        print "building the prediction model..."
        X = features.values
        y = target.values

        X_train, y_train = X, y
        # X_train, X_test, y_train, y_test = train_test_split(X, y)

        rf_grid = {'max_depth': [3, None],
                   'max_features': ['sqrt', 'log2', 10, 20, 40],
                   'min_samples_split': [1, 3, 10],
                   'min_samples_leaf': [1, 3, 10],
                   'bootstrap': [True, False],
                   'n_estimators': [25, 40, 50],
                   'random_state': [1]}

        rf_grid_search = self.grid_search(RandomForestClassifier(),
                                          rf_grid, X_train, y_train)
        rf_best = rf_grid_search.best_estimator_

        return rf_best

    def grid_search(self, est, grid, X_train, y_train):
        grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                               scoring='mean_squared_error').fit(X_train,
                                                                 y_train)
        return grid_cv

    def print_result(self):
        rf_best = self.predict_model
        X_test = self.train_features
        y_test = self.train_target
        print "score:", rf_best.score(X_test, y_test)
        y_predict = rf_best.predict(X_test)
        print "confusion matrix:"
        print confusion_matrix(y_test, y_predict)
        print "precision:", precision_score(y_test, y_predict)
        print "recall:", recall_score(y_test, y_predict)
        print "f1-score:", f1_score(y_test, y_predict)
        return


if __name__ == '__main__':

    print "starting..."

    with open('../data/patent_dataframe.pkl', 'rb') as handle:
        df_input = pkl.load(handle)

    plr = PatentLongevityPredictor()
    plr.initialization(df_input)
    plr.print_result()
    plr.predict()
