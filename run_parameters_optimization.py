# from Hybrid import HybridRecommender
#
#      dataReader = NetflixEnhancedReader()
#      URM_train = dataReader.get_URM_train()
#      URM_test = dataReader.get_URM_test()
#
#      logFile = open("BPR_MF_GridSearch.txt", "a")
#
#
#      gridSearch = GridSearch(MF_BPR_Cython, None, URM_test, None)
#
#
#      hyperparamethers_range_dictionary = {}
#      hyperparamethers_range_dictionary["num_factors"] = list(range(1, 51, 5))
#      hyperparamethers_range_dictionary["epochs"] = list(range(1, 51, 10))
#      hyperparamethers_range_dictionary["batch_size"] = list(range(1, 101, 50))
#      hyperparamethers_range_dictionary["learning_rate"] = [1e-1, 1e-2, 1e-3, 1e-4]
#
#
#
#      recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
#                               DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
#                               DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
#                               DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
#                               DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
#
#      best_paramethers = gridSearch.search(recommenderDictionary, logFile = logFile)
#
#      print(best_paramethers)
