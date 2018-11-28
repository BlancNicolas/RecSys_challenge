import pandas as pd
import scipy.sparse as sps
import numpy as np

tracks_path = "data/tracks.csv"
train_path = "data/train.csv"


class Data:
    def __init__(self):
        self.tracks_data = pd.read_csv(tracks_path)
        self.train_data = pd.read_csv(train_path)

        userList_URM = self.train_data['playlist_id'].tolist()
        itemList_URM = self.train_data['track_id'].tolist()
        ratingsList_URM = np.ones(len(self.train_data['track_id']))

        self.URM_all = sps.coo_matrix((ratingsList_URM, (userList_URM, itemList_URM)))
        self.ICM = sps.coo_matrix(self.tracks_data.values)

    def get_URM(self):
        return self.URM_all.tocsr()

    def get_ICM( self ):
        return self.ICM.tocsr()

