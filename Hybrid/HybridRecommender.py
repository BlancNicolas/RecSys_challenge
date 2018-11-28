#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np


class HybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ Hybrid recommender
    weights : list
    topK : number
    number of member : The number of recommenders involved in this one.
    """

    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, Similarities_list, sparse_weights=True):
        """
        :param URM_train:
        :param Similarities_list: The list of Similarities Matrices originating from their recommender
        :param weights_list: The list of weights we assign to each recommender
        :param sparse_weights:
        """
        super(HybridRecommender, self).__init__()

        if not all(x.shape == Similarities_list[0].shape for x in Similarities_list):
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size")

        self.Similarities_list = []

        # CSR is faster during evaluation
        for matrix in Similarities_list:
            self.Similarities_list.append(check_matrix(matrix.copy(), 'csr'))

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights

    def fit(self, topK=100, weights_list=None):
        """fit
        :param topK:
        :param weights_list: The list of weights we assign at each recommender
        :return:
        """

        if not len(self.Similarities_list) == len(weights_list):
            raise ValueError("The lists are not the same length")
        else:
            self.weights_list = weights_list

        self.topK = topK

        W = sum([np.dot(a, b) for a, b in zip(self.Similarities_list, self.weights_list)])

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)

