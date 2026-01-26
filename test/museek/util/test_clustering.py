import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from sklearn.cluster import KMeans

from museek.util.clustering import Clustering


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.step = 1.999
        # define two mock sets of timestamps with a wide gap in between
        self.mock_before_timestamps = np.arange(
            start=1638898582.8787673, stop=1638899490.4947934, step=self.step
        )
        self.mock_after_timestamps = np.arange(
            start=1638905561.9262295, stop=1638906101.697875, step=self.step
        )

    @patch.object(Clustering, "split_clusters")
    def test_ordered_dumps_of_coherent_clusters(self, mock_split_clusters):
        mock_n_clusters = Mock()
        mock_features = MagicMock()
        mock_split_clusters.return_value = [[1, 2], [5, 3]], "not used"
        ordered_dumps = Clustering().ordered_dumps_of_coherent_clusters(
            features=mock_features, n_clusters=mock_n_clusters
        )
        self.assertListEqual([1, 2], [*ordered_dumps[0]])
        self.assertListEqual([3, 4, 5], [*ordered_dumps[1]])

    def test_split_clusters(self):
        mock_timestamps = np.append(
            self.mock_before_timestamps, self.mock_after_timestamps
        )
        dumps_list, centres = Clustering().split_clusters(
            feature_vector=mock_timestamps, n_clusters=2
        )
        self.assertListEqual(list(range(0, 455)), list(dumps_list[0]))
        self.assertListEqual(list(range(455, 726)), list(dumps_list[1]))
        self.assertLess(abs(centres[0] - np.mean(self.mock_before_timestamps)), 1e-3)
        self.assertLess(abs(centres[1] - np.mean(self.mock_after_timestamps)), 1e-3)

    def test_split_clusters_when_shuffled(self):
        mock_timestamps = np.append(
            self.mock_before_timestamps, self.mock_after_timestamps
        )
        np.random.shuffle(mock_timestamps)
        dumps_list, _ = Clustering().split_clusters(
            feature_vector=mock_timestamps, n_clusters=2
        )
        for dumps in dumps_list:
            # after shuffling check which elements come first
            if mock_timestamps[dumps[0]] in self.mock_before_timestamps:
                expect = self.mock_before_timestamps
            else:
                expect = self.mock_after_timestamps
            np.testing.assert_array_equal(expect, np.sort(mock_timestamps[dumps]))

    def test_split_clusters_when_three_clusters(self):
        mock_extra_timestamps = np.arange(
            start=1648905561.9262295, stop=1648906101.697875, step=self.step
        )
        mock_timestamps = np.append(
            self.mock_before_timestamps, self.mock_after_timestamps
        )
        mock_timestamps = np.append(mock_timestamps, mock_extra_timestamps)
        (dumps_1, dumps_2, dumps_3), _ = Clustering().split_clusters(
            feature_vector=mock_timestamps, n_clusters=3
        )
        self.assertListEqual(list(range(0, 455)), list(dumps_1))
        self.assertListEqual(list(range(455, 726)), list(dumps_2))
        self.assertListEqual(list(range(726, 997)), list(dumps_3))

    def test_split_pointings(self):
        mock_right_ascension = np.asarray(
            [
                24.42243,
                24.42233,
                24.42193,
                24.42230,
                24.42196,
                24.42207,
                24.42236,
                24.42192,
                24.42239,
                24.42247,
                24.42170,
                24.42251,
                24.42183,
                24.42225,
                24.42199,
                24.42186,
                24.42248,
                24.42166,
                24.42219,
                24.42234,
                24.42174,
                24.42216,
                24.42164,
                24.42228,
                24.42177,
                24.42202,
                24.42179,
                79.95702,
                79.95681,
                79.95673,
                79.95725,
                79.95707,
                79.95695,
                79.95696,
                79.95751,
                79.95729,
                79.95765,
                79.95719,
                79.95709,
                80.67401,
                80.67413,
                80.67412,
                80.67429,
                80.67398,
                80.67467,
                79.95747,
                79.95732,
                79.95739,
                79.95729,
                79.95717,
                79.95700,
                79.95689,
                79.95713,
                79.95697,
                79.95711,
                79.95730,
                79.95724,
                79.95702,
                79.95692,
                79.95770,
                79.95736,
                79.95705,
                79.95718,
                79.24004,
                79.24041,
                79.24061,
                79.23998,
                79.24013,
                79.24025,
            ]
        )
        mock_declination = np.asarray(
            [
                33.15948,
                33.15977,
                33.15957,
                33.15962,
                33.15980,
                33.15996,
                33.15977,
                33.15996,
                33.15970,
                33.15940,
                33.15971,
                33.15970,
                33.15973,
                33.15969,
                33.15958,
                33.15968,
                33.15928,
                33.15960,
                33.15927,
                33.15977,
                33.15954,
                33.15982,
                33.15996,
                33.15974,
                33.15933,
                33.15985,
                33.15975,
                -45.27882,
                -45.27879,
                -45.27933,
                -45.27910,
                -45.27882,
                -45.27883,
                -45.77811,
                -45.77875,
                -45.77917,
                -45.77882,
                -45.77881,
                -45.77901,
                -45.77684,
                -45.77653,
                -45.77705,
                -45.77685,
                -45.77646,
                -45.77676,
                -45.77840,
                -45.77910,
                -45.77892,
                -45.77888,
                -45.77922,
                -45.77864,
                -46.27876,
                -46.27882,
                -46.27930,
                -46.27878,
                -46.27901,
                -46.27913,
                -45.77924,
                -45.77894,
                -45.77879,
                -45.77876,
                -45.77858,
                -45.77850,
                -45.77667,
                -45.77656,
                -45.77675,
                -45.77628,
                -45.77658,
                -45.77657,
            ]
        )
        mock_timestamps = np.asarray(
            [
                1638898582.87877,
                1638898596.87285,
                1638898608.86777,
                1638898618.86354,
                1638898630.85847,
                1638898640.85424,
                1638898652.84917,
                1638898666.84325,
                1638898676.83902,
                1638898688.83394,
                1638898698.82971,
                1638898710.82464,
                1638898720.82041,
                1638898732.81534,
                1638898746.80942,
                1638898756.80519,
                1638898768.80011,
                1638898778.79588,
                1638898790.79081,
                1638898802.78573,
                1638898812.78151,
                1638898826.77558,
                1638898836.77136,
                1638898848.76628,
                1638898860.76121,
                1638898870.75698,
                1638898882.75190,
                1638898958.71976,
                1638898968.71554,
                1638898982.70962,
                1638898992.70539,
                1638899004.70031,
                1638899016.69524,
                1638899038.68593,
                1638899048.68171,
                1638899062.67579,
                1638899074.67071,
                1638899084.66648,
                1638899096.66141,
                1638899116.65295,
                1638899126.64872,
                1638899140.64280,
                1638899152.63773,
                1638899162.63350,
                1638899174.62842,
                1638899194.61997,
                1638899204.61574,
                1638899218.60982,
                1638899230.60474,
                1638899240.60051,
                1638899252.59544,
                1638899272.58698,
                1638899282.58275,
                1638899296.57683,
                1638899308.57176,
                1638899318.56753,
                1638899330.56245,
                1638899350.55400,
                1638899360.54977,
                1638899374.54385,
                1638899386.53877,
                1638899396.53454,
                1638899408.52947,
                1638899430.52017,
                1638899440.51594,
                1638899454.51002,
                1638899466.50494,
                1638899476.50071,
                1638899488.49564,
            ]
        )
        pointing_dumps, pointing_centres = Clustering().split_pointings(
            coordinate_1=mock_right_ascension,
            coordinate_2=mock_declination,
            timestamps=mock_timestamps,
            n_pointings=5,
            n_centre_observations=3,
            distance_threshold=1,
        )
        centre_pointing_centre = np.array([79.95725333, -45.77881333])
        up_pointing_centre = np.asarray(
            [centre_pointing_centre[0], centre_pointing_centre[1] + 0.5]
        )  # 0.5 degree offset
        right_pointing_centre = np.asarray(
            [centre_pointing_centre[0] + 0.7, centre_pointing_centre[1]]
        )  # 0.7 degree offset
        down_pointing_centre = np.asarray(
            [centre_pointing_centre[0], centre_pointing_centre[1] - 0.5]
        )
        left_pointing_centre = np.asarray(
            [centre_pointing_centre[0] - 0.7, centre_pointing_centre[1]]
        )
        expect_centres = [
            centre_pointing_centre,
            centre_pointing_centre,
            centre_pointing_centre,
            up_pointing_centre,
            right_pointing_centre,
            down_pointing_centre,
            left_pointing_centre,
        ]
        for centre, expect_centre in zip(pointing_centres, expect_centres):
            np.testing.assert_array_almost_equal(expect_centre, centre, 1)

        for dumps, expect_centre in zip(pointing_dumps, expect_centres):
            for d in dumps:
                right_ascension_diff = mock_right_ascension[d] - expect_centre[0]
                declination_diff = mock_declination[d] - expect_centre[1]
                self.assertLess(
                    np.sum(np.abs([right_ascension_diff, declination_diff])), 0.1
                )

    @patch.object(np, "where")
    @patch.object(Clustering, "_iterative_outlier_cluster")
    @patch.object(Clustering, "_max_difference_to_mean_metric")
    def test_iterative_outlier_indices(
        self,
        mock_max_difference_to_mean_metric,
        mock_iterative_outlier_cluster,
        mock_where,
    ):
        mock_max_difference_to_mean_metric.return_value = np.array([2])
        mock_feature_vector = MagicMock()
        mock_distance_threshold = 1
        outlier_indices = Clustering().iterative_outlier_indices(
            feature_vector=mock_feature_vector,
            distance_threshold=mock_distance_threshold,
        )
        self.assertEqual(mock_where.return_value[0], outlier_indices)
        mock_iterative_outlier_cluster.assert_called_once_with(
            feature_vector=mock_feature_vector,
            n_clusters=2,
            metric=Clustering()._max_difference_to_mean_metric,
            get_outlier=Clustering()._get_outlier_cluster_binary_majority,
            distance_threshold=mock_distance_threshold,
            max_iter=5,
        )
        mock_where.assert_called_once_with(mock_iterative_outlier_cluster.return_value)

    @patch.object(Clustering, "_iterative_outlier_cluster")
    @patch.object(Clustering, "_max_difference_to_mean_metric")
    def test_iterative_outlier_indices_when_no_outliers(
        self, mock_max_difference_to_mean_metric, mock_iterative_outlier_cluster
    ):
        mock_max_difference_to_mean_metric.return_value = np.array([0.1])
        mock_feature_vector = MagicMock()
        mock_distance_threshold = 1
        outlier_indices = Clustering().iterative_outlier_indices(
            feature_vector=mock_feature_vector,
            distance_threshold=mock_distance_threshold,
        )
        self.assertListEqual([], outlier_indices)
        mock_iterative_outlier_cluster.assert_not_called()

    @patch.object(np, "atleast_2d")
    def test_atleast_2d(self, mock_atleast_2d):
        mock_array = MagicMock(shape=[1])
        result = Clustering._atleast_2d(array=mock_array)
        self.assertEqual(mock_atleast_2d.return_value.T, result)

    def test_atleast_2d_when_2d_expect_input(self):
        mock_array = MagicMock(shape=[1, 1])
        result = Clustering._atleast_2d(array=mock_array)
        self.assertEqual(mock_array, result)

    def test_ordered_and_unique(self):
        ordered = Clustering._ordered_and_unique(
            array_or_list=[11, 1, 5, 2, 3, 1, 2, 1, 4, 3, 5, 0]
        )
        self.assertListEqual([11, 1, 5, 2, 3, 4, 0], ordered)

    def test_ordered_and_unique_when_only_one_entry(self):
        ordered = Clustering._ordered_and_unique(array_or_list=[1])
        self.assertListEqual([1], ordered)

    def test_iterative_outlier_cluster_wityh_get_outlier_cluster(self):
        np.random.seed(0)
        random_1 = np.random.normal(0, scale=0.1, size=10)
        random_2 = np.random.normal(0, scale=0.1, size=10)
        centre = [[a, b] for a, b in zip(random_1, random_2)]
        up = [[a, b + 1] for a, b in zip(random_1, random_2)]
        right = [[a + 1, b] for a, b in zip(random_1, random_2)]
        down = [[a, b - 1] for a, b in zip(random_1, random_2)]
        left = [[a - 1, b] for a, b in zip(random_1, random_2)]
        outlier_1 = [[a - 10, b] for a, b in zip(random_1, random_2)]
        outlier_2 = [[a + 5, b + 5] for a, b in zip(random_1, random_2)]
        outlier_3 = [[a, b - 10] for a, b in zip(random_1, random_2)]

        mock_feature_vector = (
            centre + up + right + down + left + outlier_1 + outlier_2 + outlier_3
        )

        cluster = Clustering()._iterative_outlier_cluster(
            feature_vector=np.asarray(mock_feature_vector),
            n_clusters=5,
            metric=Clustering._separations_from_mean_metric,
            distance_threshold=2,
            get_outlier=Clustering._get_outlier_cluster,
        )
        outlier_length = len(outlier_3) + len(outlier_2) + len(outlier_1)
        for label in cluster[-outlier_length:]:
            self.assertEqual(1, label)
        for label in cluster[:-outlier_length]:
            self.assertEqual(0, label)

    def test_iterative_outlier_cluster_with_get_outlier_cluster_binary_majority(self):
        non_outliers_list = [np.random.normal(i, scale=0.1, size=10) for i in range(6)]
        outliers_list = [np.random.normal(i + 10, scale=0.1, size=9) for i in range(6)]

        non_outliers = np.asarray(non_outliers_list).T
        outliers = np.asarray(outliers_list).T
        feature_vector = np.append(non_outliers, outliers).reshape((19, 6))
        distance_threshold = 1
        cluster = Clustering()._iterative_outlier_cluster(
            feature_vector=feature_vector,
            n_clusters=2,
            metric=Clustering()._max_difference_to_mean_metric,
            get_outlier=Clustering()._get_outlier_cluster_binary_majority,
            distance_threshold=distance_threshold,
            max_iter=5,
        )
        expect = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        )
        np.testing.assert_array_equal(expect, cluster)

    @patch.object(KMeans, "fit")
    def test_get_outlier_cluster(self, mock_fit):
        mock_metric = MagicMock(return_value=[0.1, 2.0])
        mock_clusters = [0, 1]
        mock_fit.return_value.predict.return_value = mock_clusters
        mock_feature_vector = Mock()
        mock_n_clusters = Mock()
        outlier_cluster, is_done = Clustering._get_outlier_cluster(
            feature_vector=mock_feature_vector,
            n_clusters=mock_n_clusters,
            metric=mock_metric,
            distance_threshold=1,
        )
        mock_fit.assert_called_once_with(mock_feature_vector)
        np.testing.assert_array_equal(np.array([0, 1]), outlier_cluster)
        self.assertFalse(is_done)

    @patch.object(KMeans, "fit")
    def test_get_outlier_cluster_when_no_cluster(self, mock_fit):
        mock_metric = MagicMock(return_value=[0.1, 2.0])
        mock_clusters = []
        mock_fit.return_value.predict.return_value = mock_clusters
        mock_feature_vector = Mock()
        mock_n_clusters = Mock()
        outlier_cluster, _ = Clustering._get_outlier_cluster(
            feature_vector=mock_feature_vector,
            n_clusters=mock_n_clusters,
            metric=mock_metric,
            distance_threshold=0.01,
        )
        mock_fit.assert_called_once_with(mock_feature_vector)
        self.assertIsNone(outlier_cluster)

    @patch.object(KMeans, "fit")
    def test_get_outlier_cluster_when_no_outlier(self, mock_fit):
        mock_metric = MagicMock(return_value=[0.1, 2.0])
        mock_feature_vector = Mock()
        mock_n_clusters = Mock()
        outlier_cluster, _ = Clustering._get_outlier_cluster(
            feature_vector=mock_feature_vector,
            n_clusters=mock_n_clusters,
            metric=mock_metric,
            distance_threshold=3,
        )
        self.assertIsNone(outlier_cluster)
        mock_fit.return_value.predict.assert_not_called()
        mock_fit.assert_called_once_with(mock_feature_vector)

    @patch.object(KMeans, "fit")
    def test_get_outlier_cluster_when_only_one_cluster(self, mock_fit):
        mock_metric = MagicMock(return_value=[0.1, 2.0])
        mock_clusters = [1, 1]
        mock_fit.return_value.predict.return_value = mock_clusters
        mock_feature_vector = Mock()
        mock_n_clusters = Mock()
        outlier_cluster, _ = Clustering._get_outlier_cluster(
            feature_vector=mock_feature_vector,
            n_clusters=mock_n_clusters,
            metric=mock_metric,
            distance_threshold=1,
        )
        mock_fit.assert_called_once_with(mock_feature_vector)
        mock_fit.return_value.predict.assert_called_once_with(mock_feature_vector)
        self.assertIsNone(outlier_cluster)

    def test_get_outlier_cluster_binary_majority(self):
        pass

    def test_condense_nested_cluster_list(self):
        mock_cluster_list = [
            np.asarray([1, 0, 0, 0]),
            np.asarray([1, 0, 0]),
            np.asarray([1, 0]),
            np.asarray([0]),
        ]
        condensed = Clustering._condense_nested_cluster_list(
            cluster_list=mock_cluster_list
        )
        np.testing.assert_array_equal(np.array([1, 1, 1, 0]), condensed)

    def test_condense_nested_cluster_list_when_only_one_element(self):
        mock_cluster_list = [np.asarray([1, 0, 0, 0])]
        condensed = Clustering._condense_nested_cluster_list(
            cluster_list=mock_cluster_list
        )
        np.testing.assert_array_equal(np.array([1, 0, 0, 0]), condensed)

    def test_get_separations_from_mean_metric(self):
        mock_coordinates = np.array(
            [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1]]
        )
        separations = Clustering._separations_from_mean_metric(
            coordinates=mock_coordinates
        )
        np.testing.assert_array_almost_equal(
            np.array([0, 1, 1, 1, 1, np.sqrt(2), np.sqrt(2)]), separations, 4
        )

    def test_max_difference_to_mean(self):
        mock_features = np.arange(27).reshape((9, 3))
        result = Clustering()._max_difference_to_mean_metric(features=mock_features)
        np.testing.assert_array_equal(np.array([12.0, 12.0, 12.0]), result)


if __name__ == "__main__":
    unittest.main()
