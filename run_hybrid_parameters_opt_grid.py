from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from Base.Evaluation.Evaluator import SequentialEvaluator
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender_multiple_grid
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from ParameterTuning.BayesianSearch import BayesianSearch
from data.data_reader_sequential import SequentialReader
from ParameterTuning.AbstractClassSearch import DictionaryKeys
import os
import numpy as np
import itertools
from data.Movielens_10M.Movielens10MReader import split_train_validation_test
from parameters import HYBRID_ICF_CB_UCF_WEIGHTS, Fit_Parameters

data = SequentialReader()

URM = data.get_URM()
ICM = data.get_ICM()

URM_train = data.get_URM_train()
URM_test = data.get_URM_test()

print("URM_train shape : {}".format(URM_train.shape))
print("URM_test shape : {}".format(URM_test.shape))
print("ICM shape : {}".format(ICM.shape))

print("Dimensions")

print(URM_train.shape)
print(URM_test.shape)

# ------------------------
# Instanciating Evaluators
# ------------------------

evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])
#evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])

evaluator_test = EvaluatorWrapper(evaluator_test)
#evaluator_test = EvaluatorWrapper(evaluator_test)

# ------------------------
# Recommender class definition
# ------------------------

# ------------------------
# Generating lists of weights to evaluate
# ------------------------

#alpha_list = np.arange(0.1, 1.0, 0.1)
alpha_list = np.arange(0.0, 1.05, 0.05)
l1 = (alpha_list)
weight_list = list(itertools.product(l1, l1, l1))
only_sum_equal_1 = list(filter(lambda x: sum(list(x)) == 1, weight_list))
only_sum_equal_1 = list(map(lambda x : list(x), only_sum_equal_1))

# -------------------------------------------------------------
# Fitting recommender:
#
# The hybrid recommender constructor needs fitted recommenders
# -------------------------------------------------------------

UserBased = UserKNNCFRecommender(URM_train)
ItemBased = ItemKNNCFRecommender(URM_train)
ContentBased = ItemKNNCBFRecommender(ICM, URM_train)

UserBased.fit(topK=500, shrink=10)
ItemBased.fit(topK=400, shrink=50)
ContentBased.fit(topK=5, shrink=1000)

hybrid = ItemKNNScoresHybridRecommender_multiple_grid(URM_train, ItemBased, ContentBased, UserBased)

# -------------------------------
# Set path and file
# -------------------------------
output_root_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

logFile = open(output_root_path + "Hybrid_GridSearch.txt", "a")

# -------------------------------
# metric to optimize
# -------------------------------

metric_to_optimize = "map"
map_list = []
best_dict = {
    "MAP" : 0,
    "WEIGHTS" : [0,0,0],
}
i=0
for x in only_sum_equal_1:
    i += 1
    print("---- {} \%".format(i/len(only_sum_equal_1)))
    hybrid.fit(weights=x)
    res_dict = evaluator_test.evaluateRecommender(hybrid)
    map_list.append(res_dict[0][10]["MAP"])
    if res_dict[0][10]["MAP"] > best_dict["MAP"]:
        best_dict["MAP"] = res_dict[0][10]["MAP"]
        best_dict["WEIGHTS"] = x
        print("New best config : {}".format(best_dict))
        if (logFile != None):
            logFile.write("Best config: {}, Results {}\n".format(best_dict))
            logFile.flush()

print("Best config : {}".format(best_dict))


