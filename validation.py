import time
import evaluation_function as ef
from matplotlib import pyplot as plt
import multiprocessing as mult
import itertools



class Validation:
    def __init__(self, train_data, test_data, test_recommender, test_recommender_2=None):
        self.train = train_data
        self.test = test_data
        self.recommender = test_recommender
        self.optim_k = 10
        self.optim_shrink = 1
        self.MAP_parameters = []

        if test_recommender_2:
            self.recommender_2 = test_recommender_2

    def time_test(self):
        # Recommendation time evaluation
        n_users_to_test = 1000

        start_time = time.time()

        for user_id in range(n_users_to_test):
            self.recommender.recommend(user_id, at=5)

        end_time = time.time()

        print("Wrong implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))

    def parameters_tunning(self, shrink=True, neigh_nb=True):
        k_nb = [5, 10, 20, 30, 50, 100, 200, 500]
        shrink_term_nb = [0.1, 0.5, 1, 2, 4, 6, 8, 10]

        combinations = list(itertools.product(k_nb, shrink_term_nb))

        print("------------------")
        print("Tunning parameters")
        print("------------------")

        MAP_parameters = []
        for combination in combinations:
            self.recommender.fit(shrink=combination[1], k=combination[0])
            result_dict = ef.evaluate_algorithm(self.test, self.recommender)
            MAP_parameters.append(result_dict["MAP"])

        # Plot the MAP result for each k number
        plt.plot(MAP_parameters)
        plt.xlabel("Parameters combinations")
        plt.ylabel("MAP")
        plt.show()

        optim_k_index = MAP_parameters.index(max(MAP_per_k))
        self.optim_k = combinations[optim_k_index][0]
        self.optim_shrink = combinations[optim_k_index][1]

        return self.optim_shrink, self.optim_k

    def recommend_evaluate_k(self, tuple):
        if tuple:
            self.recommender.fit(shrink=tuple[1], k=tuple[0])
            result_dict = ef.evaluate_algorithm(self.test, self.recommender)
            self.MAP_parameters.append(result_dict["MAP"])
        else:
            print("No tuple given")
            return

    def parameters_tunning_parallellized( self ):
        """ parameters_tunning_paralellized use parallelized computing to tune parameters as the number
        of neighbors k or the value of the shrink parameter.
        :param shrink: If true, will be evaluated
        :param neigh_nb: If true, will be evaluated
        :return: Optim_k and Optim_shrink and it will update de value of optim_k or optim_shrink
        """

        # TODO : Verbose mode
        number_of_workers = 4
        k_nb = [5, 10, 20, 30, 50, 100, 200, 500]
        shrink_term_nb = [0.1, 0.5, 1, 2, 4, 6, 8, 10]
        combinations = list(itertools.product(k_nb, shrink_term_nb))
        self.MAP_parameters = []


        print("------------------")
        print("Tunning parameters")
        print("------------------")

        with mult.Pool(number_of_workers) as p:
            p.starmap(self.recommend_evaluate_k, k_nb)

        # Plot the MAP result for each k number
        plt.plot(self.MAP_parameters)
        plt.xlabel("Parameters combinations")
        plt.ylabel("MAP")
        plt.show()

        optim_k_index = MAP_per_k.index(max(MAP_per_k))
        self.optim_k = k_nb[optim_k_index]

        return self.optim_shrink, self.optim_k



