from copy import copy
from typing import Callable

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
        outlier_cluster = self._calibrator_pointings_outlier_cluster(feature_vector=feature,
                                                                     n_clusters=n_pointings,
                                                                     metric=self._get_separations_from_mean,
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
        on_centre_cluster = np.argmin(self._get_separations_from_mean(pointing_centres))
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

    def _calibrator_pointings_outlier_cluster(
            self,
            feature_vector: np.ndarray,
            n_clusters: int,
            metric: Callable,
            distance_threshold: float,
            max_iter: int = 10
    ) -> np.ndarray:
        """
        Return a 1-D `array` with the same length as `feature_vector` which is `1` where outliers are identified
        and zero otherwise. Outlier clusters are iteratively identified until the remainder is well contained within
        `cluster` clusters within a distance of `distance_threshold` from each other.
        :param feature_vector: feature vector for clustering
        :param n_clusters: number of clusters assumed to be relatively close to each other
        :param metric: distance metric for outlier identification
        :param distance_threshold: threshold beyond which samples are labelled outliers
        :param max_iter: maximum number of iterations in loop to ensure exit
        :return: 1-D binary array of the same length as `feature_vector`
        """
        feature_vector = self._atleast_2d(feature_vector)
        feature_vector_without_outlier = copy(feature_vector)
        outlier_cluster_list = []  # type: list[np.ndarray]
        count = 0
        while count < max_iter:
            count += 1
            outlier_cluster = self._get_outlier_cluster(feature_vector=feature_vector_without_outlier,
                                                        n_clusters=n_clusters,
                                                        metric=metric,
                                                        distance_threshold=distance_threshold)
            if outlier_cluster is not None:
                outlier_cluster_list.append(outlier_cluster)
                feature_vector_without_outlier = [feature_vector_without_outlier[i]
                                                  for i in range(len(feature_vector_without_outlier))
                                                  if outlier_cluster[i] != 1]
            else:
                break
        if outlier_cluster_list:
            return self._condense_nested_cluster_list(cluster_list=outlier_cluster_list)
        return np.zeros(feature_vector.shape[0])

    @staticmethod
    def _get_outlier_cluster(feature_vector: np.ndarray,
                             n_clusters: int,
                             metric: Callable,
                             distance_threshold: float) \
            -> np.ndarray | None:
        """
        Return optional binary array of same length as `feature` vector which is 1 where samples belong to an outlier
        cluster and zero otherwise.
        :param feature_vector: feature vector for clustering
        :param n_clusters: number of non-outlier clusters
        :param metric: distance metric for outlier identification
        :param distance_threshold: threshold beyond which samples are labelled outliers
        :return: binary 1-D array or None
        """
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(feature_vector)
        cluster_distances = metric(model.cluster_centers_)
        outlier_cluster_value = np.argmax(cluster_distances)
        if cluster_distances[outlier_cluster_value] <= distance_threshold:
            return
        clusters = model.predict(feature_vector)
        if len(np.unique(clusters)) <= 1:
            return
        result = np.zeros_like(clusters)
        result[clusters == outlier_cluster_value] = 1
        return result

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
    def _get_separations_from_mean(coordinates: np.ndarray[float]) -> np.ndarray[float]:
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
