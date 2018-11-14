import csv


def write_submission(target_users, recommender_object, path, at=10):
    """
    :param target_users: list of user we want to make a recommendation
    :param recommender_object: the recommender instance
    :param path: path for the file
    :param at: number of recommendations for each user
    :return: no return
    """

    with open(path, mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['playlist_id', 'track_ids'])

        for user_id in target_users['playlist_id']:
                recommended_items = recommender_object.recommend(user_id, at=at)
                csv_writer.writerow([user_id, ' '.join(str(x) for x in recommended_items)])
