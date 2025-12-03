import pandas as pd
import numpy as np
import os
import torch
import scipy
import random
from collections import defaultdict

TRAIN_PATH = '/kaggle/input/fmgfcb/train.csv'
TEST_PATH  = '/kaggle/input/fmgfcb/test.csv'
BOOK_GENRES_PATH = '/kaggle/input/fmgfcb/book_genres.csv'
USERS_PATH = '/kaggle/input/fmgfcb/users.csv'
SUBMISSION_PATH = 'good.csv'

C_BOOK = 15
C_USER = 7
FREQ_GAMMA = 0.094

TARGET_MEAN = 7.5
MEAN_SHIFT_ALPHA = 0.12

GENRE_ALPHA = 0.12           # общий жанровый вклад (увеличен)
TOP_N_GENRES = 10           # сколько жанров считать у пользователя (увеличено)
TOP5_GENRE_ALPHA = 0.06     # дополнительный бонус за попадание в топ-N (kept)
TOP5_GENRE_WEIGHTED = True

FAV_BOOK_TOPK = 3
FAV_BOOK_BOOST = 0.10

BOOK_POP_ALPHA = 0.02

GENDER_GENRE_ALPHA = 0.04   # дополнительная корректировка по полу пользователя

REG_G = 3.0

train = pd.read_csv(TRAIN_PATH, sep=',')
test  = pd.read_csv(TEST_PATH, sep=',')

book_genres = pd.read_csv(BOOK_GENRES_PATH, sep=',')
users = pd.read_csv(USERS_PATH, sep=',')

train_rated = train[train['has_read'] == 1].copy()
train_rated['rating'] = train_rated['rating'].astype(float)

mu_raw = train_rated['rating'].mean()
mu = mu_raw + MEAN_SHIFT_ALPHA * (TARGET_MEAN - mu_raw)

book_dev_sum = ((train_rated['rating'] - mu)).groupby(train_rated['book_id']).sum()
book_count   = train_rated.groupby('book_id')['rating'].count()
book_bias = (book_dev_sum / (book_count + C_BOOK)).to_dict()

user_dev_sum = ((train_rated['rating'] - mu)).groupby(train_rated['user_id']).sum()
user_count   = train_rated.groupby('user_id')['rating'].count()
user_bias = (user_dev_sum / (user_count + C_USER)).to_dict()

freq_user = np.log1p(user_count).to_dict()
freq_book = np.log1p(book_count).to_dict()

book_to_genres = {}
if not book_genres.empty:
    book_genres['book_id'] = book_genres['book_id'].astype(int)
    book_genres['genre_id'] = book_genres['genre_id'].astype(int)
    for b, grp in book_genres.groupby('book_id'):
        book_to_genres[int(b)] = grp['genre_id'].astype(int).tolist()

genre_global_bias = {}
if not book_genres.empty:
    tr = train_rated[['book_id', 'rating']].merge(book_genres, on='book_id', how='left')
    tr = tr[~tr['genre_id'].isna()].copy()
    if not tr.empty:
        tr['genre_id'] = tr['genre_id'].astype(int)
        tr['dev'] = tr['rating'] - mu
        g_sum = tr.groupby('genre_id')['dev'].sum()
        g_cnt = tr.groupby('genre_id')['dev'].count()
        genre_global_bias = (g_sum / (g_cnt + 1.0)).to_dict()

user_genre_bias = {}
if not book_genres.empty:
    tr2 = train_rated[['user_id', 'book_id', 'rating']].merge(book_genres, on='book_id', how='left')
    tr2 = tr2[~tr2['genre_id'].isna()].copy()
    if not tr2.empty:
        tr2['genre_id'] = tr2['genre_id'].astype(int)
        tr2['dev'] = tr2['rating'] - mu
        agg = tr2.groupby(['user_id', 'genre_id']).agg({'dev': ['mean','count']})
        agg.columns = ['dev_mean','cnt']
        agg = agg.reset_index()
        for _, row in agg.iterrows():
            user_genre_bias[(int(row['user_id']), int(row['genre_id']))] = (float(row['dev_mean']), int(row['cnt']))

user_topN_genres = {}
if user_genre_bias:
    tmp = defaultdict(list)
    for (u,g), (dev_mean, cnt) in user_genre_bias.items():
        tmp[u].append((g, dev_mean, cnt))
    for u, lst in tmp.items():
        lst_sorted = sorted(lst, key=lambda x: (x[1], x[2]), reverse=True)
        topn = [int(x[0]) for x in lst_sorted[:TOP_N_GENRES]]
        user_topN_genres[int(u)] = topn

user_gender = {}
if not users.empty and 'user_id' in users.columns and 'gender' in users.columns:
    user_gender = users.set_index('user_id')['gender'].to_dict()

gender_genre_bias = {}
if (not book_genres.empty) and (not users.empty):
    tmp = train_rated[['user_id','book_id','rating']].merge(book_genres, on='book_id', how='left')
    tmp = tmp.merge(users[['user_id','gender']], on='user_id', how='left')
    tmp = tmp[~tmp['genre_id'].isna()].copy()
    if not tmp.empty:
        tmp['dev'] = tmp['rating'] - mu
        grp = tmp.groupby(['gender','genre_id'])['dev'].agg(['sum','count']).reset_index()
        for _, row in grp.iterrows():
            gender = row['gender']
            genre = int(row['genre_id'])
            s = row['sum']
            c = row['count']
            gender_genre_bias[(int(gender) if not pd.isna(gender) else None, genre)] = float(s / (c + 1.0))

user_fav_books = {}
grp = train_rated.groupby(['user_id', 'book_id'])['rating'].agg(['mean','count']).reset_index()
for u, g in grp.groupby('user_id'):
    dfu = g.copy()
    dfu_sorted = dfu.sort_values(by=['mean','count'], ascending=[False, False])
    topk = dfu_sorted['book_id'].astype(int).tolist()[:FAV_BOOK_TOPK]
    user_fav_books[int(u)] = set(topk)

book_count_series = book_count = train_rated.groupby('book_id')['rating'].count().reindex(book_count.index if 'book_count' in locals() else book_to_genres.keys()).fillna(0).astype(float) if 'book_count' in locals() else train_rated.groupby('book_id')['rating'].count()
avg_log_count = np.log1p(book_count_series + 1.0).mean() if not book_count_series.empty else 0.0

def get_prediction(row):
    u = row['user_id']
    b = row['book_id']

    b_i = book_bias.get(b, 0.0)
    b_u = user_bias.get(u, 0.0)

    r = mu + b_u + b_i

    f_u = freq_user.get(u, 1.0)
    f_i = freq_book.get(b, 1.0)
    r += FREQ_GAMMA * (f_u / f_i)

    genres = book_to_genres.get(b, [])

    if genres:
        user_vals = []
        counts = []
        for g in genres:
            ug = user_genre_bias.get((u, g), None)
            if ug is not None:
                user_vals.append(ug[0])
                counts.append(ug[1])
        if user_vals:
            w = np.array(counts, dtype=float)
            genre_adj = float(np.average(user_vals, weights=w))
            r += GENRE_ALPHA * genre_adj
        else:
            global_vals = [genre_global_bias.get(g, 0.0) for g in genres]
            if global_vals:
                r += GENRE_ALPHA * float(np.mean(global_vals))

    topn = user_topN_genres.get(u, [])
    if genres and topn:
        overlap = set(genres).intersection(set(topn))
        frac = len(overlap) / max(1, len(genres))
        if frac > 0:
            if TOP5_GENRE_WEIGHTED:
                vals = []
                for g in overlap:
                    ug = user_genre_bias.get((u, g), None)
                    if ug is not None:
                        vals.append(ug[0])
                avg_dev = float(np.mean(vals)) if vals else 0.0
                r += TOP5_GENRE_ALPHA * frac * avg_dev
            else:
                r += TOP5_GENRE_ALPHA * frac

    gender = user_gender.get(u, None)
    if gender is not None and genres:
        gvals = []
        for g in genres:
            gg = gender_genre_bias.get((int(gender), g), None)
            if gg is not None:
                gvals.append(gg)
        if gvals:
            r += GENDER_GENRE_ALPHA * float(np.mean(gvals))

    favs = user_fav_books.get(u, set())
    if b in favs:
        r += FAV_BOOK_BOOST

    bc = freq_book.get(b, 0.0)
    pop_norm = (bc - avg_log_count) if avg_log_count is not None else 0.0
    r += BOOK_POP_ALPHA * pop_norm

    r = mu + 0.92 * (r - mu)

    return float(np.clip(r, 0.0, 10.0))

test['rating_predict'] = test.apply(get_prediction, axis=1)
test[['user_id', 'book_id', 'rating_predict']].to_csv(SUBMISSION_PATH, index=False)
