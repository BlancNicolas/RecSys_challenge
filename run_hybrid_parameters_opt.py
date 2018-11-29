from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from Base.Evaluation.Evaluator import SequentialEvaluator
from data.Data_formating import Data
from data_splitter import train_test_holdout
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender_multiple
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from ParameterTuning.BayesianSearch import BayesianSearch
from ParameterTuning.AbstractClassSearch import DictionaryKeys
import os
import numpy as np
import itertools
from data.Movielens_10M.Movielens10MReader import split_train_validation_test
from parameters import HYBRID_ICF_CB_UCF_WEIGHTS, Fit_Parameters

data = Data()
URM = data.get_URM()
ICM = data.get_ICM()

URM_train, URM_validation, URM_test = split_train_validation_test(URM, [0.8, 0.1, 0.1])

print("Dimensions")

print(URM_train.shape)
print(URM_validation.shape)
print(URM_test.shape)

# ------------------------
# Instanciating Evaluators
# ------------------------

evaluator_validation = SequentialEvaluator(URM_validation, cutoff_list=[10])
evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])

evaluator_validation = EvaluatorWrapper(evaluator_validation)
evaluator_test = EvaluatorWrapper(evaluator_test)

# ------------------------
# Recommender class definition
# ------------------------

recommender_class = ItemKNNScoresHybridRecommender_multiple

# ------------------------
# Instanciating BayesianSearch
# ------------------------
parameterSearch = BayesianSearch(recommender_class,
                                 evaluator_validation=evaluator_validation,
                                 evaluator_test=evaluator_test)

# ------------------------
# Generating lists of weights to evaluate
# ------------------------

#alpha_list = np.arange(0.1, 1.0, 0.1)
alpha_list = np.arange(0.0, 1.05, 0.05)
l1 = (alpha_list)
weight_list = list(itertools.product(l1, l1, l1))
only_sum_equal_1 = list(filter(lambda x: sum(list(x)) == 1, weight_list))
only_sum_equal_1 = list(map(lambda x : list(x), only_sum_equal_1))
list_weight_1 = list(map(lambda x : x[0], only_sum_equal_1))
list_weight_2 = list(map(lambda x : x[1], only_sum_equal_1))
list_weight_3 = list(map(lambda x : x[2], only_sum_equal_1))

# -------------------------------
# Defining parameters dictionnary
# -------------------------------

hyperparamethers_range_dictionary = {}
hyperparamethers_range_dictionary["weight_1"] = list_weight_1
hyperparamethers_range_dictionary["weight_2"] = list_weight_2
hyperparamethers_range_dictionary["weight_3"] = list_weight_3


# -------------------------------------------------------------
# Fitting recommender:
#
# The hybrid recommender constructor needs fitted recommenders
# -------------------------------------------------------------

UserBased = UserKNNCFRecommender(URM_train)
ItemBased = ItemKNNCFRecommender(URM_train)
ContentBased = ItemKNNCBFRecommender(ICM, URM_train)

UserBased.fit(topK=Fit_Parameters.UCF_TOPK, shrink=Fit_Parameters.UCF_SHRINK)
ItemBased.fit(topK=Fit_Parameters.ICF_TOPK, shrink=Fit_Parameters.ICF_SHRINK)
ContentBased.fit(topK=Fit_Parameters.CB_TOPK, shrink=Fit_Parameters.CB_SHRINK)


# -------------------------------
# Instanciating dictionnary
# -------------------------------

recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ItemBased, ContentBased, UserBased],
                         DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                         DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                         DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                         DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

# -------------------------------
# Set path and file
# -------------------------------
output_root_path = "result_experiments/Hybrid_opt.txt"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

# -------------------------------
# n_cases, metric to optimize
# -------------------------------

n_cases = 10
metric_to_optimize = "MAP"

best_parameters = parameterSearch.search(recommenderDictionary,
                                         n_cases = n_cases,
                                         output_root_path = output_root_path,
                                         metric=metric_to_optimize,
                                         init_points=20)