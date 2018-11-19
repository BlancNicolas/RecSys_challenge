import numpy as np

from Compute_Similarity_Python import Compute_Similarity_Python
from Base.Recommender_utils import check_matrix

from Base.Recommender import Recommender
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from Base.cosine_similarity import Cosine_Similarity


class TopPopRecommender(object):

    def fit(self, URM_train):

        self.URM_train = URM_train

        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=5, remove_seen=True):

        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items


class ItemCBFKNNRecommender(object):

    def __init__( self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit( self, k=50, shrink=100, normalize=True, similarity="cosine" ):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=k, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend( self, user_id, at=None, exclude_seen=True ):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen( self, user_id, scores ):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


class ItemKNNCFRecommender(Recommender, Similarity_Matrix_Recommender):
    """ ItemKNN recommender"""

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

    def fit(self, k=50, shrink=100, similarity='cosine', normalize=True):

        self.k = k
        self.shrink = shrink

        self.similarity = Cosine_Similarity(self.URM_train, shrink=shrink, topK=k, normalize=normalize, mode = similarity)

        if self.sparse_weights:
            self.W_sparse = self.similarity.compute_similarity()
        else:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()

class HybridRecommender(Recommender, Similarity_Matrix_Recommender):

    def __init__(self, URM ,URM_train, recommender_1, recommender_2, sparse_weights=True):
        super(HybridRecommender, self).__init__()

        self.URM = URM
        self.URM_train = check_matrix(URM_train, 'csr')
        self.recommender_CB = recommender_1
        self.recommender_CF = recommender_2
        self.sparse_weights = sparse_weights

    def fit( self, k_1=5, shrink_1=0.5, k_2=500, shrink_2=10, similarity='cosine', normalize=True ):

        self.k_CB = k_1
        self.k_CF = k_2
        self.shrink_CB= shrink_1
        self.shrink_CF = shrink_2

        self.recommender_CB.fit(shrink=self.shrink_CB, k=self.k_CB)
        self.recommender_CF.fit(shrink=self.shrink_CF, k=self.k_CF)

    def recommend( self, user_id, at=None, exclude_seen=True, weight_CB=0.1, weight_CF=0.9):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]

        scores_CB = user_profile.dot(self.recommender_CB.W_sparse).toarray().ravel()
        scores_CF = user_profile.dot(self.recommender_CF.W_sparse).toarray().ravel()

        # use weights
        scores_CB = scores_CB * weight_CB
        scores_CF = scores_CF * weight_CF

        self.scores = scores_CB + scores_CF

        if exclude_seen:
            scores = self.filter_seen(user_id, self.scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen( self, user_id, scores ):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

