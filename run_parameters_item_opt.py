from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from Base.Evaluation.Evaluator import SequentialEvaluator
from data.Data_formating import Data
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from ParameterTuning.BayesianSearch import BayesianSearch
from data.data_reader_sequential import SequentialReader
from ParameterTuning.AbstractClassSearch import DictionaryKeys
from data.Movielens_10M.Movielens10MReader import split_train_validation_test

data = SequentialReader()

URM = data.get_URM()
ICM = data.get_ICM()

URM_train = data.get_URM_train()
URM_test = data.get_URM_test()

print("URM_train shape : {}".format(URM_train.shape))
print("URM_test shape : {}".format(URM_test.shape))
print("ICM shape : {}".format(ICM.shape))


# ------------------------
# Instanciating Evaluators
# ------------------------

evaluator_validation = SequentialEvaluator(URM_test, cutoff_list=[10])
#evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])

evaluator_validation = EvaluatorWrapper(evaluator_validation)
#evaluator_test = EvaluatorWrapper(evaluator_test)

# ------------------------
# Recommender class definition
# ------------------------

recommender_class = ItemKNNCFRecommender

# ------------------------
# Instanciating BayesianSearch
# ------------------------

parameterSearch = BayesianSearch(recommender_class,
                                 evaluator_validation=evaluator_validation)

# -------------------------------
# Defining parameters dictionnary
# -------------------------------


hyperparamethers_range_dictionary = {}
hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
hyperparamethers_range_dictionary["similarity"] = ["cosine"]
hyperparamethers_range_dictionary["normalize"] = [True, False]

recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                         DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                         DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                         DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                         DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

# -------------------------------
# Set path and file
# -------------------------------

output_root_path = "result_experiments/Item-based_opt.txt"

n_cases = 20
metric_to_optimize = "MAP"

best_parameters = parameterSearch.search(recommenderDictionary,
                                         n_cases = n_cases,
                                         output_root_path = output_root_path,
                                         metric=metric_to_optimize,
                                         init_points=15)