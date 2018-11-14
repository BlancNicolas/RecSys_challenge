import time

class Validation:
    def __init__( self, train_data, test_data, test_recommender):
        self.train = train_data
        self.test = test_data
        self.recommender = test_recommender


    def time_test( self ):
        # Recommendation time evaluation
        n_users_to_test = 1000

        start_time = time.time()

        for user_id in range(n_users_to_test):
            self.recommender.recommend(user_id, at=5)

        end_time = time.time()

        print("Wrong implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))

    def parameters_tunning( self, shrink=None, neigh_nb=None):
        
        if shrink == True:


