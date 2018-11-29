from data.Data_formating import Data
from data.Movielens_10M.Movielens10MReader import split_train_validation_test
from Base.Evaluation.Evaluator import SequentialEvaluator
from data.data_reader_sequential import SequentialReader
from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender_multiple
from parameters import HYBRID_ICF_CB_UCF_WEIGHTS, Fit_Parameters
import write_submission as ws
import pandas as pd

data = SequentialReader()

URM = data.get_URM()
ICM = data.get_ICM()

URM_train = data.get_URM_train()
URM_test = data.get_URM_test()

print("URM_train shape : {}".format(URM_train.shape))
print("URM_test shape : {}".format(URM_test.shape))
print("ICM shape : {}".format(ICM.shape))

evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])
evaluator_test = EvaluatorWrapper(evaluator_test)

UserBased = UserKNNCFRecommender(URM_train)
UserBased.fit(topK=500, shrink=10)

ContentBased = ItemKNNCBFRecommender(ICM, URM_train)
ContentBased.fit(topK=5, shrink=1000)

ItemKNNCF = ItemKNNCFRecommender(URM_train)
ItemKNNCF.fit(topK=400, shrink=50)

hybridRecommender_scores = ItemKNNScoresHybridRecommender_multiple(URM_train, ItemKNNCF, ContentBased, UserBased)
hybridRecommender_scores.fit(weight_1=HYBRID_ICF_CB_UCF_WEIGHTS[0]
                             , weight_2=HYBRID_ICF_CB_UCF_WEIGHTS[1]
                             , weight_3=HYBRID_ICF_CB_UCF_WEIGHTS[2])

dict, _ = evaluator_test.evaluateRecommender(hybridRecommender_scores)

print(dict)

target_data = pd.read_csv('data/target_playlists.csv')

ws.write_submission(target_data, hybridRecommender_scores, 'output/submission.csv', at=10)
