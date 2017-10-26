# https://www.kaggle.com/kamilkk/simple-fast-lgbm-0-66683
# copied 2017-10-12

import gc
import logging
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

start_time = time.time()
# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

logger.debug('Loading data...')
data_path = './input/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv', parse_dates=['registration_init_time', 'expiration_date'])

logger.debug('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

# todo figure out why dropping the expiration date here makes our AUC worse

if True:
    # https://www.kaggle.com/juanumusic/days-instead-of-dates-lgbm-0-66870
    members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype(int)
else:
    # todo fix this
    members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
    members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

    # todo fix this
    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
    members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time', 'expiration_date'], axis=1)

members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

del members, songs
gc.collect()

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

del train, test
gc.collect()

random_state = 1
test_size = 0.2
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)

del X, y
gc.collect()

d_train = lgb.Dataset(X_train, label=y_train, feature_name=cols, categorical_feature = 'auto')
d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=cols, categorical_feature = 'auto')

logger.debug('Training LGBM model...')
params = {'learning_rate': 0.4, 'application': 'binary',
          'max_depth': -1, # was 15
          'num_leaves': 2 ** 8, 'verbosity': 0,
          'metric': 'auc'}

num_boost_rounds = 1000
early_stopping_rounds = 20
cv_nfold = 5
logger.debug('cross-validating with %d folds' % cv_nfold)
evaluation_history = lgb.cv(params, train_set=d_train, num_boost_round=num_boost_rounds, nfold=cv_nfold,
                            early_stopping_rounds=early_stopping_rounds)
logger.debug(evaluation_history)
logger.debug(evaluation_history.keys())
actual_boost_rounds = len(evaluation_history['auc-mean'])
logger.debug('training the model with %d rounds' % actual_boost_rounds)

watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=actual_boost_rounds, valid_sets=watchlist,
                  early_stopping_rounds=early_stopping_rounds, verbose_eval=10)

logger.debug('Making predictions.')
p_test = model.predict(X_test)

logger.debug('writing the submission file')
submission_data = pd.DataFrame()
submission_data['id'] = ids
submission_data['target'] = p_test
submission_data.to_csv('submission.csv.gz', compression='gzip', index=False, float_format='%.5f')

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)

feature_file = 'lightgb_features.png'
logger.debug('Plotting feature importance %s' % feature_file)
ax = lgb.plot_importance(model)
plt.savefig('./' + feature_file)
