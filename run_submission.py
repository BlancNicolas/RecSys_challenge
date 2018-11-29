from data.Data_formating import Data
from data.Movielens_10M.Movielens10MReader import split_train_validation_test
from Base.Evaluation.Evaluator import SequentialEvaluator
from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender_multiple
from parameters import HYBRID_ICF_CB_UCF_WEIGHTS, Fit_Parameters

data = Data()
URM = data.get_URM()
ICM = data.get_ICM()

URM_train, _, URM_test = split_train_validation_test(URM, [0.8, 0.2, 0.0])
evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])
evaluator_test = EvaluatorWrapper(evaluator_test)

UserBased = UserKNNCFRecommender(URM_train)
UserBased.fit(topK=150, shrink=0)

ContentBased = ItemKNNCBFRecommender(ICM, URM_train)
ContentBased.fit(topK=50, shrink=100)

ItemKNNCF = ItemKNNCFRecommender(URM_train)
ItemKNNCF.fit(topK=200, shrink=50)

hybridRecommender_scores = ItemKNNScoresHybridRecommender_multiple(URM_train, ItemKNNCF, ContentBased, UserBased)
hybridRecommender_scores.fit(weight_1=HYBRID_ICF_CB_UCF_WEIGHTS[0]
                             , weight_2=HYBRID_ICF_CB_UCF_WEIGHTS[1]
                             , weight_3=HYBRID_ICF_CB_UCF_WEIGHTS[2])

dict, _ = evaluator_test.evaluateRecommender(hybridRecommender_scores)

print(dict)
