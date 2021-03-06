from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender


class ItemKNNScoresHybridRecommender(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__( self, URM_train, Recommender_1, Recommender_2 ):
        super(ItemKNNScoresHybridRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

        self.compute_item_score = self.compute_score_hybrid

    def fit( self, alpha=0.5 ):
        self.alpha = alpha

    def compute_score_hybrid( self, user_id_array ):
        print("len user_id_array : {}".format(user_id_array))
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights

class ItemKNNScoresHybridRecommender_multiple(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender_multiple"

    def __init__( self, URM_train, Recommender_1, Recommender_2, Recommender_3 ):
        super(ItemKNNScoresHybridRecommender_multiple, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.weight_1 = 0
        self.weight_2 = 0
        self.weight_3 = 0

        self.compute_item_score = self.compute_score_hybrid

    def fit(self, weight_1=None,  weight_2=None,  weight_3=None):
        if weight_1 and weight_2 and weight_3:
            self.weight_1 = weight_1
            self.weight_2 = weight_2
            self.weight_3 = weight_3

    def compute_score_hybrid( self, user_id_array):
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3.compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.weight_1 + item_weights_2 * self.weight_2 \
                       + item_weights_3 * self.weight_3

        return item_weights

    def saveModel( self, folder_path, file_name=None ):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "Weight_1": self.weight_1,
            "Weight_2": self.weight_2,
            "Weight_2": self.weight_3,
            "Recommenders_orders": self.Recommender_1.RECOMMENDER_NAME + " - " + self.Recommender_2.RECOMMENDER_NAME  + " - " +self.Recommender_3.RECOMMENDER_NAME
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")

class ItemKNNScoresHybridRecommender_multiple_grid(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender_multiple_grid"

    def __init__( self, URM_train, Recommender_1, Recommender_2, Recommender_3 ):
        super(ItemKNNScoresHybridRecommender_multiple_grid, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.weights = []

        self.compute_item_score = self.compute_score_hybrid

    def fit(self, weights=None):
        if weights:
            self.weights = weights

    def compute_score_hybrid( self, user_id_array):

        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3.compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.weights[0] + item_weights_2 * self.weights[1] \
                       + item_weights_3 * self.weights[2]

        return item_weights

    def saveModel( self, folder_path, file_name=None ):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "Weight_1": self.weight_1,
            "Weight_2": self.weight_2,
            "Weight_2": self.weight_3,
            "Recommenders_orders": self.Recommender_1.RECOMMENDER_NAME + " - " + self.Recommender_2.RECOMMENDER_NAME  + " - " +self.Recommender_3.RECOMMENDER_NAME
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")

