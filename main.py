import pandas as pd
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import evaluation_function as ef

from data_splitter import train_test_holdout
from recommenders import ItemCBFKNNRecommender

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
URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

# Create recommender
recommender = ItemCBFKNNRecommender(URM_train, ICM)
recommender.fit(shrink=0.0, topK=30)


# Recommendation time evaluation
n_users_to_test = 1000

start_time = time.time()

for user_id in range(n_users_to_test):
    recommender.recommend(user_id, at=5)

end_time = time.time()

print("Wrong implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))

# Evaluation of neighbors

neigh_nb = [5, 10, 20, 30, 50, 100, 200, 500]
MAP_per_k = []
MAP_per_shrink = []

for topK in neigh_nb:

    recommender.fit(shrink=0.0, topK=topK)
    result_dict = ef.evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

shrink_term = [0.1, 0.5, 1, 2, 4, 6, 8, 10]

for shrink in shrink_term:

    recommender.fit(shrink=shrink, )
    result_dict = ef.evaluate_algorithm(URM_test, recommender)
    MAP_per_shrink.append(result_dict["MAP"])


