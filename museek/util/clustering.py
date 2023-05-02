from copy import copy
from typing import Callable, Any

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from sklearn.cluster import KMeans


class Clustering:
    """
    Clustering related class to split samples.
    One use-case is to split a set of calibrator observations into on-centre and the different off-centre parts,
    i.e. up, down, right and left. The split can also be done time-wise, i.e. split at a significant time gap.
    """

    def ordered_dumps_of_coherent_clusters(
            self,
            features: np.ndarray,
            n_clusters: int
    ) -> list[range, range]:
        """
        Assume that clustered samples are next to each other inside `features` and return a list of the `range`s
        from first to last index of `n_clusters` clusters.
        """
        cluster_indices_list, _ = self.split_clusters(feature_vector=features,
                                                      n_clusters=n_clusters)
        dumps = [range(min(indices), max(indices) + 1) for indices in cluster_indices_list]
        return dumps

    def split_clusters(self, feature_vector: np.ndarray, n_clusters: int) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Return a `list` of indices belonging to each of `n_clusters` clusters in `feature_vector`.
        The return `list` is ordered according to the clusters' first appearances in `feature_vector`.
        Uses the `KMeans` algorithm.
        """
        feature = self._atleast_2d(array=feature_vector)
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(feature)
        clusters = model.predict(feature)

        ordered_cluster_labels = self._ordered_and_unique(array_or_list=clusters)
        cluster_centres = model.cluster_centers_[ordered_cluster_labels]

        cluster_indices_list = [np.where(clusters == cluster_label)[0] for cluster_label in ordered_cluster_labels]

        return cluster_indices_list, cluster_centres

    def split_pointings(
            self,
            coordinate_1: np.ndarray,
            coordinate_2: np.ndarray,
            timestamps: np.ndarray,
            n_pointings: int,
            n_centre_observations: int,
            distance_threshold: float
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Return a tuple of a list of dumps for each calibrator pointing
        and the corresponding pointing centre coordinates.
        :param coordinate_1: first coordinate, will be part of feature vector
        :param coordinate_2: second coordinate, will be part of feature vector
        :param timestamps: array containing the dump timestamps, necessary to split up the on-centre pointings
        :param n_pointings: number of pointing directions, usually 5
        :param n_centre_observations: number of target on-centre observations
        :param distance_threshold: pointings beyond this threshold are considered outliers
        :return: a tuple, first entry is a list of dump indices, second entry a 2-d array containing coordinates
        """
        feature = np.asarray([[a, b] for a, b in zip(coordinate_1, coordinate_2)])
        outlier_cluster = self._iterative_outlier_cluster(feature_vector=feature,
                                                          n_clusters=n_pointings,
                                                          metric=self._separations_from_mean_metric,
                                                          get_outlier=self._get_outlier_cluster,
                                                          distance_threshold=distance_threshold)
        no_outlier_indices = np.asarray([i for i, out in enumerate(outlier_cluster) if out != 1])

        pointing_indices, pointing_centres = self.split_clusters(feature_vector=feature[no_outlier_indices],
                                                                 n_clusters=n_pointings)
        # re-insert the outlier indices
        pointing_dumps = [no_outlier_indices[pointing] for pointing in pointing_indices]

        centre_right_ascension = [c[0] for c in pointing_centres]
        centre_declination = [c[1] for c in pointing_centres]

        off_centre_order = [
            np.argmax(centre_declination),
            np.argmax(centre_right_ascension),
            np.argmin(centre_declination),
            np.argmin(centre_right_ascension)
        ]  # clock-wise top, right, down, left
        on_centre_cluster = np.argmin(self._separations_from_mean_metric(pointing_centres))
        all_on_centre_dumps = pointing_dumps[on_centre_cluster]
        split_the_centre_dumps = self.ordered_dumps_of_coherent_clusters(
            features=timestamps[all_on_centre_dumps],
            n_clusters=n_centre_observations
        )
        on_centre_dumps = [all_on_centre_dumps[d] for d in split_the_centre_dumps]
        split_centre_centres = np.asarray([pointing_centres[on_centre_cluster]] * 3)
        pointing_dumps = on_centre_dumps + [pointing_dumps[o] for o in off_centre_order]
        pointing_centres = np.append(split_centre_centres, pointing_centres[off_centre_order], axis=0)

        return pointing_dumps, pointing_centres

    def iterative_outlier_indices(self, feature_vector: np.ndarray, distance_threshold: float) -> list[int]:
        """
        Return the indices of outlier samples in `feature_vector` `(n_samples, n_features)`
        imposing that non-outliers should be contained within an separation of `distance_threshold`.
        """
        metric = self._max_difference_to_mean_metric
        if any(metric(features=feature_vector) > distance_threshold):
            outlier_cluster = self._iterative_outlier_cluster(feature_vector=feature_vector,
                                                              n_clusters=2,
                                                              metric=metric,
                                                              get_outlier=self._get_outlier_cluster_binary_majority,
                                                              distance_threshold=distance_threshold,
                                                              max_iter=5)
            return np.where(outlier_cluster)[0]
        return []

    @staticmethod
    def _atleast_2d(array: np.ndarray) -> np.ndarray:
        """ Return the transpose of `np.atleast_2d(array)` if `array` is 1-dimensional, else `array` is returned. """
        if len(array.shape) == 1:
            feature = np.atleast_2d(array).T
        else:
            feature = array
        return feature

    @staticmethod
    def _ordered_and_unique(array_or_list: list[int] | np.ndarray[int]) -> list[int]:
        """ Return unique integers in 1-D `np.ndarray` or `list` `array` in the order they first appear. """
        ordered = []
        for i in array_or_list:
            if i not in ordered:
                ordered.append(i)
        return ordered

    def _iterative_outlier_cluster(
            self,
            feature_vector: np.ndarray,
            n_clusters: int,
            metric: Callable,
            get_outlier: Callable,
            distance_threshold: float,
            max_iter: int = 10
    ) -> np.ndarray:
        """
        Return a 1-D `array` with the same length as `feature_vector` which is `True` where outliers are identified
        and zero otherwise. Outlier clusters are iteratively identified until the remainder is well contained within
        `cluster` clusters within a distance of `distance_threshold` from each other.
        :param feature_vector: feature vector for clustering `(n_samples, n_features)`
        :param n_clusters: number of clusters assumed to be relatively close to each other
        :param metric: distance metric for outlier identification
        :param get_outlier: `Callable` with specific signature that returns an optional boolean 1-D array of length
                            `n_samples` identifying outliers and `boolean` which finishes the iteration if `True`. If
                            the `array` is `None`, the iteration is stopped immediately
        :param distance_threshold: threshold beyond which samples are labelled outliers
        :param max_iter: maximum number of iterations in loop to ensure exit
        :return: 1-D boolean array of the same length as `n_samples`
        """
        feature_vector = self._atleast_2d(feature_vector)
        feature_vector_without_outlier = copy(feature_vector)
        outlier_cluster_list = []  # type: list[np.ndarray]
        count = 0
        while count < max_iter:
            count += 1
            outlier_cluster, is_done = get_outlier(feature_vector=feature_vector_without_outlier,
                                                   n_clusters=n_clusters,
                                                   metric=metric,
                                                   distance_threshold=distance_threshold)
            if outlier_cluster is None:
                break
            if is_done:
                count = max_iter
            outlier_cluster_list.append(outlier_cluster)
            feature_vector_without_outlier = np.asarray([feature_vector_without_outlier[i]
                                                         for i in range(len(feature_vector_without_outlier))
                                                         if outlier_cluster[i] != 1])
        if outlier_cluster_list:
            return self._condense_nested_cluster_list(cluster_list=outlier_cluster_list)
        return np.zeros(feature_vector.shape[0], dtype=bool)

    @staticmethod
    def _get_outlier_cluster(feature_vector: np.ndarray,
                             n_clusters: int,
                             metric: Callable,
                             distance_threshold: float) \
            -> tuple[np.ndarray | None, bool]:
        """
        Return optional binary array of same length as `feature` vector which is 1 where samples belong to an outlier
        cluster and zero otherwise and `False`.
        :param feature_vector: feature vector for clustering `(n_samples, n_features)`
        :param n_clusters: number of non-outlier clusters
        :param metric: distance metric for outlier identification
        :param distance_threshold: threshold beyond which samples are labelled outliers
        :return: `tuple` of optional 1-D array and `False`
        """
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(feature_vector)
        cluster_distances = metric(model.cluster_centers_)
        outlier_cluster_value = np.argmax(cluster_distances)
        if cluster_distances[outlier_cluster_value] <= distance_threshold:
            return None, False
        clusters = model.predict(feature_vector)
        if len(np.unique(clusters)) <= 1:
            return None, False
        result = np.zeros_like(clusters, dtype=bool)
        result[clusters == outlier_cluster_value] = True
        return result, False

    @staticmethod
    def _get_outlier_cluster_binary_majority(feature_vector: np.ndarray,
                                             n_clusters: Any,
                                             distance_threshold: float,
                                             metric: Callable) \
            -> tuple[np.ndarray | None, bool]:
        """
        Return `tuple` of optional binary array of same length as `feature` vector which is 1 where samples belong
        to outliers and zero otherwise and a `boolean` which is `True` if `metric` returns values
        smaller than `distance_threshold`.
        :param feature_vector: feature vector for clustering `(n_samples, n_features)`
        :param n_clusters: unused
        :param distance_threshold: if the return of `metric` is smaller than this `True` is returned to
                                   imply that this methods needs to run no further
        :param metric: `function` to calculate distances in `feature_vector`
        :return: `tuple` of binary 1-D array and `boolean`
        """
        is_done = False
        clusters = KMeans(n_clusters=2, random_state=0, n_init='auto').fit_predict(feature_vector)
        if len(np.unique(clusters)) <= 1:
            return None, is_done
        outlier_cluster_value = 1
        if sum(clusters) / len(clusters) > 0.5:
            outlier_cluster_value = 0
        inliers = np.where(clusters != outlier_cluster_value)[0]
        if len(inliers) == 0:
            return None, is_done
        distance = metric(feature_vector[inliers])
        if distance is not None and all(distance < distance_threshold):
            is_done = True
        result = np.zeros_like(clusters, dtype=bool)
        result[clusters == outlier_cluster_value] = True
        return result, is_done

    @staticmethod
    def _condense_nested_cluster_list(cluster_list: list[np.ndarray]) -> np.ndarray:
        """
        Return binary 1-D array combining all input clusters. Assumes that each element in `cluster_list` except the
        first one contains clustering information with respect to the previous element's zero entries.
        :param cluster_list: list of binary 1-D arrays contianing clustering information, each element needs
                             to have the same length as the previous elements has elements equal to zero
        :return: binary 1-D array combining all input clusters
        """
        result = cluster_list[0]
        for outlier_cluster in cluster_list[1:]:
            labels_are_zero = np.where(result == 0)[0]
            labels_are_one = labels_are_zero[np.where(outlier_cluster == 1)[0]]
            result[labels_are_one] = 1
        return result

    @staticmethod
    def _separations_from_mean_metric(coordinates: np.ndarray[float]) -> np.ndarray[float]:
        """
        Return the separation of each element in `coordinates` from the mean of `coordinates`.
        :param coordinates: `numpy` array with two dimensions, first axis for samples,
                            second for coordinate one and two
        :return: 1-D `numpy` array with separations of each point from the mean of all
        """
        mean = SkyCoord(*np.mean(coordinates, axis=0) * units.deg, frame='icrs')
        coordinates = SkyCoord(coordinates * units.deg, frame='icrs')
        separations = coordinates.separation(mean)
        return separations.degree

    @staticmethod
    def _max_difference_to_mean_metric(features: np.ndarray) -> np.ndarray[float]:
        """
        Return the maximum difference of each feature in `features` to the mean of each feature
        :param features: feature vector `(n_samples, n_features)`
        :return: vector of maximum differences with length `n_features`
        """
        return np.max(abs(features - np.mean(features, axis=0)), axis=0)
