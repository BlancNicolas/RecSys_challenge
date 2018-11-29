import numpy as np
from data.Data_formating import Data
from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from Base.Evaluation.Evaluator import SequentialEvaluator
import write_submission as ws
from GraphBased.P3alphaRecommender import P3alphaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNScoresHybridRecommender import  ItemKNNScoresHybridRecommender, ItemKNNScoresHybridRecommender_multiple
from MatrixFactorization.PureSVD import PureSVDRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from ParameterTuning.AbstractClassSearch import writeLog
from data_splitter import train_test_holdout
import itertools


data = Data()
URM = data.get_URM()
ICM = data.get_ICM()

URM_train, URM_test = train_test_holdout(URM, train_perc=0.8)

print("Shape : {}".format(URM_train.__sizeof__()))

evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])
evaluator_test = EvaluatorWrapper(evaluator_test)

output_root_path = "result_experiments/"

filename = P3alphaRecommender.RECOMMENDER_NAME \
           + ItemKNNCBFRecommender.RECOMMENDER_NAME \
           + ItemKNNCFRecommender.RECOMMENDER_NAME \
           + "hybrid_opt"

output_root_path += filename
output_file = open(output_root_path, "a")

P3alpha = P3alphaRecommender(URM_train)
P3alpha.fit(topK=100, alpha=0.7905462550621185, implicit=True, normalize_similarity=True)
# print("-------------------")
# print("--P3alpha fitted---")
# print("-------------------")

UserBased = UserKNNCFRecommender(URM_train)
UserBased.fit(topK=300, shrink=200)

ContentBased = ItemKNNCBFRecommender(ICM, URM_train)
ContentBased.fit(topK=50, shrink=100)
# print("-------------------")
# print("--KNNCBF fitted---")
# print("-------------------")
ItemKNNCF = ItemKNNCFRecommender(URM_train)
ItemKNNCF.fit(topK=300, shrink=100)
# print("-------------------")
# print("---KNNCF fitted----")
# print("-------------------")
PureSVD = PureSVDRecommender(URM_train)
PureSVD.fit(num_factors=240)
# print("-------------------")
# print("---PureSVD fitted--")
# print("-------------------")

#hybridRecommender = ItemKNNSimilarityHybridRecommender(URM_train, ItemKNNCF.W_sparse, ContentBased.W_sparse)
hybridRecommender_scores = ItemKNNScoresHybridRecommender_multiple(URM_train, ItemKNNCF, ContentBased, UserBased)
#hybridRecommender_scores = ItemKNNScoresHybridRecommender(URM_train, ItemKNNCF, PureSVD)

alpha_list = np.arange(0.1, 1.0, 0.1)
l1 = (alpha_list)
weight_list = list(itertools.product(l1, l1, l1))
only_sum_equal_1 = list(filter(lambda x: sum(list(x)) == 1, weight_list))


for x in only_sum_equal_1:
    print("-------------------")
    print("---weights = {}---".format(x))
    print("-------------------")
    hybridRecommender_scores.fit(weights=x)
    # print("-------------------")
    # print("---Hybrid fitted---")
    # print("-------------------")

    print("-------------------")
    print("-Hybrid Evaluation-")
    print("-------------------")

    dict, _ = evaluator_test.evaluateRecommender(hybridRecommender_scores)

    writeLog("---weights = {}---".format(x), output_file)
    writeLog("--- Parameters : {} ".format(dict), output_file)
    print(dict)





# target_data = pd.read_csv('data/target_playlists.csv')
#
# print("--------------------------")
# print("------Recommendation------")
# print("--------------------------")
# ws.write_submission(target_data, , 'output/submission.csv', at=10)

#print(hybrid_recommender.evaluateRecommendations(URM_test))
