
import pandas as pd
import scipy.sparse as sps
import random
import numpy as np

tracks_path = "data/tracks.csv"
train_path = "data/train.csv"
train_data_sequential_path = "data/train_sequential.csv"
target_data_path = "data/target_playlists.csv"

def random_split(lis, percent_train=0.8):
    random.shuffle(lis)
    nb_elements_train = int(percent_train*len(lis))
    nb_elem_test = int((len(lis) - nb_elements_train)/2)

    return lis[:nb_elements_train], lis[nb_elements_train:nb_elements_train+nb_elem_test], lis[nb_elements_train+nb_elem_test:]


class SequentialReaderValidation:


    def __init__(self):



        self.tracks_data = pd.read_csv(tracks_path)
        self.train_data = pd.read_csv(train_path)
        self.train_data_sequential = pd.read_csv(train_data_sequential_path)
        self.target_data = pd.read_csv(target_data_path)


        ## ORDERED PLAYLISTS
        # Take first 80% of ordered playlists for train
        # Take last 20% of ordered playlists for test

        playlists_ratings_sequential = self.train_data_sequential

        playlists_ratings_sequential = playlists_ratings_sequential.groupby('playlist_id').agg(list) \
             .applymap(lambda list: (list[:int(0.8*len(list))], list[int(0.8*len(list)):int(0.9*len(list))], list[int(0.9*len(list)):])).reset_index()

        playlists_ratings_sequential[['train_list', 'validation_list', 'test_list']] = playlists_ratings_sequential['track_id'].apply(
            pd.Series)

        # TRAIN

        playlists_ratings_sequential_train = playlists_ratings_sequential[['playlist_id', 'train_list']]
        playlists_ratings_sequential_train = playlists_ratings_sequential_train.train_list.apply(pd.Series) \
            .merge(playlists_ratings_sequential_train, right_index=True, left_index=True) \
            .drop(["train_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)

        # TEST

        playlists_ratings_sequential_test = playlists_ratings_sequential[['playlist_id', 'test_list']]
        playlists_ratings_sequential_test = playlists_ratings_sequential_test.test_list.apply(pd.Series) \
            .merge(playlists_ratings_sequential_test, right_index=True, left_index=True) \
            .drop(["test_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)

        # VALIDATION

        playlists_ratings_sequential_validation = playlists_ratings_sequential[['playlist_id', 'validation_list']]
        playlists_ratings_sequential_validation = playlists_ratings_sequential_validation.validation_list.apply(pd.Series) \
            .merge(playlists_ratings_sequential_validation, right_index=True, left_index=True) \
            .drop(["validation_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)




        ## RANDOM PLAYLISTS

        playlists_ratings_random = pd.concat([self.train_data, self.train_data_sequential]).drop_duplicates(keep=False)

        playlists_ratings_random = playlists_ratings_random.groupby(['playlist_id']).agg(list) \
            .applymap(lambda list: random_split(list, 0.8)).reset_index()

        playlists_ratings_random[['train_list','validation_list', 'test_list']] = playlists_ratings_random['track_id'].apply(
            pd.Series)

        # TRAIN

        playlists_ratings_random_train = playlists_ratings_random[['playlist_id', 'train_list']]
        playlists_ratings_random_train = playlists_ratings_random_train.train_list.apply(pd.Series) \
            .merge(playlists_ratings_random_train, right_index=True, left_index=True) \
            .drop(["train_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)

        # TEST

        playlists_ratings_random_test = playlists_ratings_random[['playlist_id', 'test_list']]
        playlists_ratings_random_test = playlists_ratings_random_test.test_list.apply(pd.Series) \
            .merge(playlists_ratings_random_test, right_index=True, left_index=True) \
            .drop(["test_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)

        # VALIDATION

        playlists_ratings_random_validation = playlists_ratings_random[['playlist_id', 'validation_list']]
        playlists_ratings_random_validation = playlists_ratings_random_validation.validation_list.apply(pd.Series) \
            .merge(playlists_ratings_random_validation, right_index=True, left_index=True) \
            .drop(["validation_list"], axis=1) \
            .melt(id_vars=['playlist_id'], value_name="track_id") \
            .drop("variable", axis=1) \
            .dropna().applymap(int)


        self.train_set = pd.concat([playlists_ratings_sequential_train, playlists_ratings_random_train], ignore_index=True)
        self.test_set = pd.concat([playlists_ratings_sequential_test, playlists_ratings_random_test], ignore_index=True)
        self.validation_set = pd.concat([playlists_ratings_sequential_validation, playlists_ratings_random_validation], ignore_index=True)
        print(len(self.train_set))
        print(len(self.test_set))
        print(len(self.validation_set))

        # URM_all
        userList_URM = self.train_data['playlist_id'].tolist()
        itemList_URM = self.train_data['track_id'].tolist()
        ratingsList_URM = np.ones(len(self.train_data['track_id']))
        self.URM_all = sps.coo_matrix((ratingsList_URM, (userList_URM, itemList_URM)))

        # ICM
        self.ICM_all = sps.coo_matrix(self.tracks_data.values)
        self.ICM_all = self.ICM_all.tocsr()

        # URM_train
        train_userList_URM = self.train_set['playlist_id'].tolist()
        train_itemList_URM = self.train_set['track_id'].tolist()
        train_ratingsList_URM = np.ones(len(self.train_set['track_id']))
        self.URM_train = sps.coo_matrix((train_ratingsList_URM, (train_userList_URM, train_itemList_URM)),
                                        shape=self.URM_all.shape)

        # URM_test
        test_userList_URM = self.test_set['playlist_id'].tolist()
        test_itemList_URM = self.test_set['track_id'].tolist()
        test_ratingsList_URM = np.ones(len(self.test_set['track_id']))
        self.URM_test = sps.coo_matrix((test_ratingsList_URM, (test_userList_URM, test_itemList_URM)),
                                       shape=self.URM_all.shape)



        # URM_validation
        validation_userList_URM = self.validation_set['playlist_id'].tolist()
        validation_itemList_URM = self.validation_set['track_id'].tolist()
        validation_ratingsList_URM = np.ones(len(self.validation_set['track_id']))
        self.URM_validation = sps.coo_matrix((validation_ratingsList_URM, (validation_userList_URM, validation_itemList_URM)),
                                       shape=self.URM_all.shape)





    def get_URM(self):
        return self.URM_all.tocsr()

    def get_ICM(self):
        return self.ICM_all.tocsr()

    def get_URM_test(self):
        return self.URM_test.tocsr()

    def get_URM_train(self):
        return self.URM_train.tocsr()

    def get_URM_validation(self):
        return self.URM_validation.tocsr()




