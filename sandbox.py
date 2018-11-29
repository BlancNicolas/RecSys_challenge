from data.Data_formating import Data
from recommenders import ItemKNNCFRecommender, ItemCBFKNNRecommender, HybridRecommender
from data_splitter import train_test_holdout
from evaluation_function import evaluate_algorithm

data = Data()
URM = data.get_URM()
ICM = data.get_ICM()
URM_train, URM_test = train_test_holdout(URM)


# OLD METHOD
contentBased = ItemCBFKNNRecommender(URM_train, ICM)
itemBased = ItemKNNCFRecommender(URM_train)
hybrid = HybridRecommender(URM, URM_train, contentBased, itemBased)
hybrid.fit()

# itemBased = ItemKNNCFRecommender(URM_train)
# itemBased.fit()
result = evaluate_algorithm(URM_test, itemBased, at=10)
print(result)

# NEW METHOD
