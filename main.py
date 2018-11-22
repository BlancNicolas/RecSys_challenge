import pandas as pd
import time

import numpy as np
import pandas as pd
from KNN import ItemKNNCFRecommender
import scipy.sparse as sps
import write_submission as ws
import validation

from data_splitter import train_test_holdout
#from recommenders import TopPopRecommender, ItemKNNCFRecommender, HybridRecommender, ItemCBFKNNRecommender
from KNN import ItemKNNCFRecommender
import run_parameters_optimization as rpo
from ParameterTuning import BayesianSearch
from Base.Evaluation.Evaluator import ParallelEvaluator

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

# recommender_CB = ItemCBFKNNRecommender(URM_train, ICM)
# recommender_CF = ItemKNNCFRecommender(URM_train)
# hybrid_recommender = HybridRecommender(URM_all, URM_train, recommender_CB, recommender_CF)
# hybrid_recommender.fit()
Evaluator = ParallelEvaluator(URM_test,[10])
Recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train=URM_train)
Bayesian = BayesianSearch.BayesianSearch(Recommender, Evaluator)
rpo.runParameterSearch_Collaborative(recommender_class=Recommender, URM_train=URM_train, metric_to_optimize="MAP", evaluator_validation=Evaluator)

# # ef.evaluate_algorithm(URM_test, topPopRecommender)
#
# # Evaluation of algorithm
# validator = validation.Validation(URM_train, URM_test, recommender)
# optim_shrink, optim_k = validator.parameters_tunning()
#
# print("Optimum k : {}".format(optim_k))
# print("Optimum shrink : {}".format(optim_shrink))
#
# recommender.fit(shrink=optim_shrink, k=optim_k)

#target_data = pd.read_csv('data/target_playlists.csv')

#ws.write_submission(target_data, hybrid_recommender, 'output/submission.csv', at=10)

#print(hybrid_recommender.evaluateRecommendations(URM_test))
