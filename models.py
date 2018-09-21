from surprise import KNNBasic, SVD
from collections import defaultdict
from operator import itemgetter
import heapq


class KnnUB(KNNBasic):
    """KNNBasic Model extension
    It extend the KNNBasic model to add the
    get_top_n_recommendations method
    configuration is based on User Based Collaborative Filtering
    
    Arguments:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.    
    """


    def __init__(self, k=40, min_k=1, verbose=True, **kwargs):
        device_based_options = {'name': 'cosine',
                                'user_based': True
                                }
        super().__init__(k=k, min_k=min_k,
                         sim_options=device_based_options, verbose=verbose, **kwargs)

    def get_top_n_recommendations(self, device_id, n: int = 10) -> list:
        """Get the top n Recomendation skills for a given device id

        Arguments:
            device_id {string} -- The id of the device

        Keyword Arguments:
            n {int} -- Top n skills to recommend (default: {10})

        Raises:
            ValueError -- Raise if the device is not on the data
            or if the model has not been trained before calling this method

        Returns:
            list -- A list of tuple containing the skills id and rate sum
        """

        try:
            self.trainset
            self.sim
        except AttributeError:
            raise ValueError(
                'Model needs to be trained first try running the fit method first')

        self.compute_similarities()
        # Get top N similar devices to our device
        device_inner_id = self.trainset.to_inner_uid(device_id)
        similarity_row = self.sim[device_inner_id]

        similar_users = []
        for inner_id, score in enumerate(similarity_row):
            if (inner_id != device_inner_id):
                similar_users.append((inner_id, score))
        k_neighbors = heapq.nlargest(n, similar_users, key=lambda t: t[1])

        # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for similar_user in k_neighbors:
            inner_id = similar_user[0]
            user_similarity_score = similar_user[1]
            their_ratings = self.trainset.ur[inner_id]
            for rating in their_ratings:
                candidates[rating[0]] += (rating[1] / 5.0) * \
                    user_similarity_score

        # Build a dictionary of stuff the devices has already use
        watched = {}
        for item_id, rating in self.trainset.ur[device_inner_id]:
            watched[item_id] = 1

        # Get top-rated skills from similar devices:
        pos = 0
        recommendations = []
        for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not item_id in watched:
                skill_id = self.trainset.to_raw_iid(item_id)
                recommendations.append((skill_id, rating_sum))
                pos += 1
                if (pos > n):
                    break

        return recommendations


class KnnIB(KNNBasic):
    """KNNBasic Model extension
    It extend the KNNBasic model to add the
    get_top_n_recommendations method
    configuration is based on Item-Based Collaborative Filtering
    
    Arguments:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.    
    """
    def __init__(self, k=40, min_k=1, verbose=True, **kwargs):
        device_based_options = {'name': 'cosine',
                                'user_based': False
                                }
        super().__init__(k=k, min_k=min_k,
                         sim_options=device_based_options, verbose=verbose, **kwargs)

    def get_top_n_recommendations(self, device_id, n: int = 10) -> list:
        """Get the top n Recomendation skills for a given device id

        Arguments:
            device_id {string} -- The id of the device

        Keyword Arguments:
            n {int} -- Top n skills to recommend (default: {10})

        Raises:
            ValueError -- Raise if the device is not on the data
            or if the model has not been trained before calling this method

        Returns:
            list -- A list of tuple containing the skills id and rate sum
        """

        try:
            self.trainset
            self.sim
        except AttributeError:
            raise ValueError(
                'Model needs to be trained first try running the fit method first')

        self.compute_similarities()
        device_inner_id = self.trainset.to_inner_uid(device_id)

        # Get the top K items we rated
        test_device_ratings = self.trainset.ur[device_inner_id]
        k_neighbors = heapq.nlargest(
            n, test_device_ratings, key=lambda t: t[1])

        # Get similar items to stuff we liked (weighted by rating)
        candidates = defaultdict(float)
        for item_id, rating in k_neighbors:
            similarity_row = self.sim[item_id]
            for inner_id, score in enumerate(similarity_row):
                candidates[inner_id] += score * (rating / 5.0)

        # Build a dictionary of stuff the user has already seen
        watched = {}
        for item_id, rating in self.trainset.ur[device_inner_id]:
            watched[item_id] = 1

        recommendations = []
        pos = 0
        for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not item_id in watched:
                skill_id = self.trainset.to_raw_iid(item_id)
                recommendations.append((skill_id, rating_sum))
                pos += 1
                if (pos > n):
                    break

        return recommendations


class SVDRecommender(SVD):
    """Extension of the SVD Model
    it add get_top_n_recommendations
    to an SVD Model once trained this model
    is able to generate recommendations with more
    diversity than the others
    """

    def __init__(self):
        super().__init__()

    def get_top_n_recommendations(self, device_id, n: int = 10) -> list:

        try:
            self.trainset
        except AttributeError:
            raise ValueError(
                'Model needs to be trained first try running the fit method first')
        
        trainset = self.trainset
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(device_id)
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]

        predictions = self.test(anti_testset)

        recommendations = []
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            recommendations.append((movieID, estimatedRating))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]
