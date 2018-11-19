import pandas as pd
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import write_submission as ws
import validation

from data_splitter import train_test_holdout
from recommenders import ItemCBFKNNRecommender
from recommenders import TopPopRecommender, ItemKNNCFRecommender

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
URM_train, URM_validation = train_test_holdout(URM_all, train_perc=0.9)

# Create recommender
recommender = ItemCBFKNNRecommender(URM_train, ICM)

# topPopRecommender = TopPopRecommender()
# topPopRecommender.fit(URM_train)

# ef.evaluate_algorithm(URM_test, topPopRecommender)

# Evaluation of algorithm
validator = validation.Validation(URM_train, URM_validation, URM_test, recommender)
optim_shrink, optim_k = validator.parameters_tunning()

print("Optimum k : {}".format(optim_k))
print("Optimum shrink : {}".format(optim_shrink))

recommender.fit(shrink=optim_shrink, topK=optim_k)

target_data = pd.read_csv('data/target_playlists.csv')

ws.write_submission(target_data, recommender, 'output/submission.csv', at=10)

myrecommender = ItemKNNCFRecommender(URM_train)
myrecommender.fit(shrink=100, k=50)
print(myrecommender.evaluateRecommendations(URM_test))
