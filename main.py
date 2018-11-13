import pandas as pd
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import evaluation_function as ef
import write_submission as ws

from data_splitter import train_test_holdout
from recommenders import TopPopRecommender

# Loading data
tracks_data = pd.read_csv("data/tracks.csv")
train_data = pd.read_csv("data/train.csv")

# Split data for sparse matrices
itemList = list(range(0, len(tracks_data)))
itemFeatureList = list(tracks_data.columns)
itemContentList = tracks_data.values.tolist()

tracksList_all = list(tracks_data.loc[:, 'track_id'])
tracksList_rated = list(train_data.loc[:, 'track_id'])


userList_URM = train_data['playlist_id'].tolist()
itemList_URM = train_data['track_id'].tolist()
ratingsList_URM = np.ones(len(train_data['track_id']))
# Create URM

URM_all = sps.coo_matrix((ratingsList_URM, (userList_URM, itemList_URM)))
URM_all = URM_all.tocsr()


# Create Train et Test sets
URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

# Create recommender
topPopRecommender = TopPopRecommender()
topPopRecommender.fit(URM_train)

#ef.evaluate_algorithm(URM_test, topPopRecommender)

target_data = pd.read_csv('data/target_playlists.csv')

ws.write_submission(target_data, topPopRecommender, 'output/submission.csv', at=10)
