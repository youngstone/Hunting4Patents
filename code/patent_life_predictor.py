import pandas as pd
import numpy as np
import re
import cPickle as pkl
import json
from collections import Counter
from datetime import datetime, date

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import cPickle as pkl
from nltk.stem.snowball import SnowballStemmer
import re
from patent_tokenizer import PatentTokenizer

import numpy as np
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
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    fig = plt.figure(figsize=(12, 8))

    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


class PatentLongevityPredictor(object):

    def __init__(self):
        self.df = None
        self.features = None
        self.target = None
        self.predict_model = None
        self.vectorizer = None
        self.test_prediction = None

    def initialization(self, file_name):
        df_input = pd.read_pickle(file_name)
        self.df = df_input

        df_transformed = self.transform(self.df)
        
        self.df_train = df_transformed[df_transformed['Patent Status'] != -1]
        self.df_test = df_transformed[df_transformed['Patent Status'] == -1]

        self.train_features = 
        self.train_target = 

        self.test_features = 
        # self.test_prediction = None

        best_estimator_ = self.build_model(self.features, self.target)

        self.predict_model = best_estimator_

        return

    def fit(self):

        pass

    # def transform(self, vectorizer):
    #     self.vectorizer = vectorizer


    #     self.X_train = 
    #     self.y_train = 
    #     self.
    #     pass

    def predict(self, df_test):
        df_test = df_test
        self.X_test = 
        pass

    def transform(self, df):
        df = df
        cond = df['Filing Date'].apply(lambda x: type(x)) == pd.tslib.NaTType
        df = df[~cond]

        cod = df['legal-events'].apply(get_code)
        dat = df['legal-events'].apply(get_date)
        fil_dat = df['Filing Date']

        end = []
        for d, e in zip(fil_dat, cod):
            end.append(determine_early_termination(d, e))

        patent_status = pd.Series(data=end)
        df['Patent Status'] = patent_status

        
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



        return df_**


    def build_model(self, features, target):
        X = features
        y = target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

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

    def grid_search(est, grid, X_train, y_train):
        grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                            scoring='mean_squared_error').fit(X_train, y_train)
        return grid_cv

    # below are some utility functions

    def events_count(self, entry):
    code = []
    if type(entry) != float:
        for event in entry:
            code.append(event['Code'])
    return Counter(code)

    def get_code(self, entry):
        code = []
        if type(entry) != float:
            for event in entry:
                code.append(event['Code'])
        return code

    def get_date(self, entry):
        date = []
        if type(entry) != float:
            for event in entry:
                dt = event['Date']
                dt_dt = datetime.strptime(dt, '%b %d, %Y')
                date.append(dt_dt)
        return date

    def determine_early_termination(self, fil_dat, code):
        today = date.today() 
        screening_day = datetime(today.year - 20, today.month, today.day)
        if fil_dat.to_datetime() < screening_day:
            return 0
        elif 'FP' in code:
            return 1
        else:
            return -1




################




# In[15]:



# In[16]:



# In[17]:




# In[18]:




# In[19]:

screened_df = screened_df[feature_names]
screened_df = screened_df.rename(columns={'Patent Number': 'Number'})

screened_df['num_of_claims'] = screened_df['Claims'].apply(len)

fwd_cit_ratio_inv_exm = []
fwd_cit_inv = screened_df['forward-citations-by-inventor']
fwd_cit_exm = screened_df['forward-citations-by-examiner']
fwd_cit_ratio_inv_exm = [float(i) / (i + e) for i, e in zip(fwd_cit_inv, fwd_cit_exm)]
screened_df['fwd_cit_ratio_inv_exm'] = fwd_cit_ratio_inv_exm
screened_df['fwd_cit_ratio_inv_exm'] = screened_df['fwd_cit_ratio_inv_exm'].fillna(float(1))


bwd_cit_ratio_inv_exm = []
bwd_cit_inv = screened_df['backward-citations-by-inventor']
bwd_cit_exm = screened_df['backward-citations-by-examiner']
bwd_cit_ratio_inv_exm = [float(i) / (i + e) for i, e in zip(bwd_cit_inv, bwd_cit_exm)]
screened_df['bwd_cit_ratio_inv_exm'] = bwd_cit_ratio_inv_exm
screened_df['bwd_cit_ratio_inv_exm'] = screened_df['bwd_cit_ratio_inv_exm'].fillna(float(1))

screened_df['bwd_pat_cit_count'] = screened_df['backward-citations'].apply(len)
screened_df['Primary Class'] = screened_df['Primary Class'].apply(lambda x: x[0])
screened_df['npl-citations'] = screened_df['npl-citations'].fillna(0)

screened_df['Claims'] = screened_df['Claims'].apply(lambda x: ' '.join(x))


drop_columns = ['US Patent References',
                'forward-citations-by-inventor', 
                'forward-citations-by-examiner',
                'backward-citations-by-inventor', 
                'backward-citations-by-examiner',
                'backward-citations']

df_model = screened_df.drop(drop_columns, axis=1)



save_path = '../data/df_model.pkl'
df_model.to_pickle(save_path)

patent_tokenizer = PatentTokenizer()
patent_tokenizer.set_df(save_path)
patent_tokenizer.set_vectors()

patent_title_vectorizer = patent_tokenizer.get_title_vectorizer()
patent_abstract_vectorizer = patent_tokenizer.get_abstract_vectorizer()
patent_claims_vectorizer = patent_tokenizer.get_claims_vectorizer()

patent_title_vectors = patent_tokenizer.get_title_vectors()
patent_abstract_vectors = patent_tokenizer.get_abstract_vectors()
patent_claims_vectors = patent_tokenizer.get_claims_vectors()



claims_features_name = patent_claims_vectorizer.get_feature_names()

df_claims_features = pd.DataFrame(data=patent_claims_vectors, index=df_model.index, columns=claims_features_name)

df_merge = pd.merge(df_model, df_claims_features, how='inner', left_index=True, right_index=True)



target = df_merge['Patent Status']


non_needed = ['Number', 'Title', 'Abstract', 'Claims', 'Patent Status']
features = df_merge.drop(non_needed, axis=1)





# In[40]:

# cols_v = get_dummies(df['venue_country'], dummy_na=True, prefix='venue_country')
dummy_class = get_dummies(features['Primary Class'], dummy_na=True, prefix='primary_class')


# In[41]:

df_rf_features = pd.merge(features, dummy_class, how='inner', left_index=True, right_index=True)


# In[42]:

df_rf_features = df_rf_features.drop(['Primary Class'], axis=1)


# In[43]:

df_rf_features < 0


# In[44]:



# In[45]:

y = target.values
X = df_rf_features.values








print "score:", rf_best.score(X_test, y_test)
y_predict = rf_best.predict(X_test)
print "confusion matrix:"
print confusion_matrix(y_test, y_predict)
print "precision:", precision_score(y_test, y_predict)
print "recall:", recall_score(y_test, y_predict)



rf_best.feature_importances_


# In[65]:

importances = rf_best.feature_importances_[:n]
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(n):
    print("%d. %s (%f)" % (f + 1, df_rf_features.columns[indices[f]], importances[indices[f]]))


def load_dataframe():
    print "loading dataframe..."
    with open('../data/patent_dataframe.pkl', 'rb') as handle:
        df = pkl.load(handle)
    return df

if __name__ == '__main__':
    pass



