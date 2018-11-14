import pandas as pd
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import evaluation_function as ef
import write_submission as ws

from data_splitter import train_test_holdout
from recommenders import ItemCBFKNNRecommender
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

# Create ICM
ICM = sps.coo_matrix(tracks_data.values)
ICM = ICM.tocsr()

# Create Train et Test sets
URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.9)

# Create recommender
recommender = ItemCBFKNNRecommender(URM_train, ICM)

# topPopRecommender = TopPopRecommender()
# topPopRecommender.fit(URM_train)

# ef.evaluate_algorithm(URM_test, topPopRecommender)

# Evaluation of neighbors

k_nb = [5, 10, 20, 30, 50, 100, 200, 500]
MAP_per_k = []
MAP_per_shrink = []

for topK in k_nb:
    recommender.fit(shrink=0.0, topK=topK)
    result_dict = ef.evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

optim_k_index = MAP_per_k.index(max(MAP_per_k))
optim_k = k_nb[optim_k_index]

shrink_term_nb = [0.1, 0.5, 1, 2, 4, 6, 8, 10]

for shrink in shrink_term_nb:
    recommender.fit(shrink=shrink, topK=optim_k)
    result_dict = ef.evaluate_algorithm(URM_test, recommender)
    MAP_per_shrink.append(result_dict["MAP"])


optim_shrink_index = MAP_per_shrink.index(max(MAP_per_shrink))
optim_shrink = shrink_term_nb[optim_shrink_index]

print("Optimum k : {}".format(optim_k))
print("Optimum shrink : {}".format(optim_shrink))

recommender.fit(shrink=optim_shrink, topK=optim_k)

target_data = pd.read_csv('data/target_playlists.csv')

ws.write_submission(target_data, recommender, 'output/submission.csv', at=10)