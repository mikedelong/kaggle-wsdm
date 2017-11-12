# https://www.kaggle.com/kamilkk/simple-fast-lgbm-0-6685
import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import gc
import logging
import time

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

# print('Loading data...')
data_path = './input/'

# train = pd.read_csv(data_path + 'train.csv', dtype={'msno': 'category',
#                                                     'source_system_tab': 'category',
#                                                     'source_screen_name': 'category',
#                                                     'source_type': 'category',
#                                                     'target': np.uint8,
#                                                     'song_id': 'category'})
# test = pd.read_csv(data_path + 'test.csv', dtype={'msno': 'category',
#                                                   'source_system_tab': 'category',
#                                                   'source_screen_name': 'category',
#                                                   'source_type': 'category',
#                                                   'song_id': 'category'})
# songs = pd.read_csv(data_path + 'songs.csv', dtype={'genre_ids': 'category',
#                                                     'language': 'category',
#                                                     'artist_name': 'category',
#                                                     'composer': 'category',
#                                                     'lyricist': 'category',
#                                                     'song_id': 'category'})
# members = pd.read_csv(data_path + 'members.csv', dtype={'city': 'category',
#                                                         'bd': np.uint8,
#                                                         'gender': 'category',
#                                                         'registered_via': 'category'})
# songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
#
# print('Data preprocessing...')
# song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
# train = train.merge(songs[song_cols], on='song_id', how='left')
# test = test.merge(songs[song_cols], on='song_id', how='left')
#
# members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
# members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
# members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))
#
# members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
# members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
# members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
# members = members.drop(['registration_init_time'], axis=1)
#
#
# def isrc_to_year(isrc):
#     if type(isrc) == str:
#         if int(isrc[5:7]) > 17:
#             return 1900 + int(isrc[5:7])
#         else:
#             return 2000 + int(isrc[5:7])
#     else:
#         return np.nan
#
#
# songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
# songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)
#
# train = train.merge(members, on='msno', how='left')
# test = test.merge(members, on='msno', how='left')
#
# train = train.merge(songs_extra, on='song_id', how='left')
# test = test.merge(songs_extra, on='song_id', how='left')
#
#
# del members, songs
# gc.collect()
#
# for col in train.columns:
#     if train[col].dtype == object:
#         train[col] = train[col].astype('category')
#         test[col] = test[col].astype('category')
#
# X = train.drop(['target'], axis=1)
# y = train['target'].values
#
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
#
# del train, test
# gc.collect()
#
# d_train = lgb.Dataset(X, y)
# watchlist = [d_train]
#
# # todo vary parameters
# print('Training LGBM model...')
# params = {'application': 'binary', 'learning_rate': 0.2, 'max_depth': 8, 'metric': 'auc', 'num_leaves': 2 ** 8,
#           'verbosity': 0}
#
# model = lgb.train(params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, verbose_eval=5)
#
# print('Making predictions and saving them...')
# p_test = model.predict(X_test)
#
# subm = pd.DataFrame()
# subm['id'] = ids
# subm['target'] = p_test
# subm.to_csv('submission.csv.gz', compression='gzip', index=False, float_format='%.5f')
# print('Done!')

logger.debug('Loading data...')
# data_path = '../input/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')

logger.debug('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

# import gc

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

random_seed = 1
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=random_seed)

del X, y
gc.collect()

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)

watchlist = [d_train, d_valid]

logger.debug('Training LGBM model...')
params = {'application': 'binary', 'learning_rate': 0.4, 'max_depth': 15, 'metric': 'auc', 'num_leaves': 2 ** 8,
          'verbosity': 0}

model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=20,
                  verbose_eval=10)

logger.debug('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression='gzip', index=False, float_format='%.5f')

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
