import time
import evaluation_function as ef
from matplotlib import pyplot as plt


class Validation:
    def __init__(self, train_data, validation_data, test_data, test_recommender):
        self.train = train_data
        self.validation = validation_data
        self.test = test_data
        self.recommender = test_recommender
        self.optim_k = 10
        self.optim_shrink = 1

    def time_test(self):
        # Recommendation time evaluation
        n_users_to_test = 1000

        start_time = time.time()

        for user_id in range(n_users_to_test):
            self.recommender.recommend(user_id, at=5)

        end_time = time.time()

        print("Wrong implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))

    def parameters_tunning(self, shrink=True, neigh_nb=True):

        if neigh_nb:

            print("------------------")
            print("Tunning of k neighbors number")
            print("------------------")

            k_nb = [5, 10, 20, 30, 50, 100, 200, 500]
            MAP_per_k = []
            for k in k_nb:
                self.recommender.fit(shrink=self.optim_shrink, topK=k)
                result_dict = ef.evaluate_algorithm(self.validation, self.recommender)
                MAP_per_k.append(result_dict["MAP"])

            # Plot the MAP result for each k number
            plt.plot(MAP_per_k)
            plt.xlabel("k number")
            plt.ylabel("MAP")
            plt.show()

            optim_k_index = MAP_per_k.index(max(MAP_per_k))
            self.optim_k = k_nb[optim_k_index]

        if shrink:
            print("------------------")
            print("Tunning of shrink")
            print("------------------")

            shrink_term_nb = [0.1, 0.5, 1, 2, 4, 6, 8, 10]
            MAP_per_shrink = []
            for shrink in shrink_term_nb:
                self.recommender.fit(shrink=shrink, topK=self.optim_k)
                result_dict = ef.evaluate_algorithm(self.validation, self.recommender)
                MAP_per_shrink.append(result_dict["MAP"])

            # Plot the MAP result for each k number
            plt.plot(MAP_per_shrink)
            plt.xlabel("Shrink value")
            plt.ylabel("MAP")
            plt.show()

            optim_shrink_index = MAP_per_shrink.index(max(MAP_per_shrink))
            self.optim_shrink = shrink_term_nb[optim_shrink_index]

        return self.optim_shrink, self.optim_k



