import unittest

import numpy as np

from museek.util.clustering import Clustering


class TestClustering(unittest.TestCase):

    def setUp(self):
        self.step = 1.999
        # define two mock sets of timestamps with a wide gap in between
        self.mock_before_timestamps = np.arange(start=1638898582.8787673, stop=1638899490.4947934, step=self.step)
        self.mock_after_timestamps = np.arange(start=1638905561.9262295, stop=1638906101.697875, step=self.step)

    def test_ordered_dumps_of_coherent_clusters(self):
        pass

    def test_split_clusters(self):
        mock_timestamps = np.append(self.mock_before_timestamps, self.mock_after_timestamps)
        dumps_list, _ = Clustering().split_clusters(feature_vector=mock_timestamps, n_clusters=2)
        self.assertListEqual(list(range(0, 455)), list(dumps_list[0]))
        self.assertListEqual(list(range(455, 726)), list(dumps_list[1]))

    def test_split_clusters_when_shuffled(self):
        mock_timestamps = np.append(self.mock_before_timestamps, self.mock_after_timestamps)
        np.random.shuffle(mock_timestamps)
        dumps_list, _ = Clustering().split_clusters(feature_vector=mock_timestamps, n_clusters=2)
        for dumps in dumps_list:
            # after shuffling check which elements come first
            if mock_timestamps[dumps[0]] in self.mock_before_timestamps:
                expect = self.mock_before_timestamps
            else:
                expect = self.mock_after_timestamps
            np.testing.assert_array_equal(expect, np.sort(mock_timestamps[dumps]))

    def test_split_clusters_when_three_clusters(self):
        mock_extra_timestamps = np.arange(start=1648905561.9262295, stop=1648906101.697875, step=self.step)
        mock_timestamps = np.append(self.mock_before_timestamps, self.mock_after_timestamps)
        mock_timestamps = np.append(mock_timestamps, mock_extra_timestamps)
        (dumps_1, dumps_2, dumps_3), _ = Clustering().split_clusters(feature_vector=mock_timestamps, n_clusters=3)
        self.assertListEqual(list(range(0, 455)), list(dumps_1))
        self.assertListEqual(list(range(455, 726)), list(dumps_2))
        self.assertListEqual(list(range(726, 997)), list(dumps_3))

    def test_split_clusters_expect_cluster_centres_correct(self):
        pass

    def test_split_pointings(self):
        mock_right_ascension = np.asarray([
            24.42243,
            24.42196,
            24.42239,
            24.42183,
            24.42248,
            24.42174,
            24.42177,
            79.95681,
            79.95695,
            79.95765,
            80.67413,
            80.67467,
            79.95729,
            79.95713,
            79.95724,
            79.95736,
            79.24041,
            79.24025,
            79.95786,
            79.95824,
            79.95640,
            80.67432,
            79.95721,
            79.95749,
            79.95747,
            79.95706,
            79.95785,
            79.23990
        ])
        mock_declination = np.asarray([
            33.15948,
            33.15980,
            33.15970,
            33.15973,
            33.15928,
            33.15954,
            33.15933,
            -45.27879,
            -45.27883,
            -45.77882,
            -45.77653,
            -45.77676,
            -45.77888,
            -46.27882,
            -46.27913,
            -45.77876,
            -45.77656,
            -45.77657,
            -45.27913,
            -45.77827,
            -45.77924,
            -45.77641,
            -45.77905,
            -45.77878,
            -46.27891,
            -45.77891,
            -45.77838,
            -45.77806
        ])
        mock_timestamps = np.asarray([
            1638898582.87877,
            1638898630.85847,
            1638898676.83902,
            1638898720.82041,
            1638898768.80011,
            1638898812.78151,
            1638898860.76121,
            1638898968.71554,
            1638899016.69524,
            1638899074.67071,
            1638899126.64872,
            1638899174.62842,
            1638899230.60474,
            1638899282.58275,
            1638899330.56245,
            1638899386.53877,
            1638899440.51594,
            1638899488.49564,
            1638905599.91016,
            1638905657.88563,
            1638905705.86533,
            1638905761.84165,
            1638905811.82051,
            1638905859.80021,
            1638905909.77907,
            1638905963.75623,
            1638906009.73678,
            1638906061.71479
        ])
        pointing_dumps, pointing_centres = Clustering().split_pointings(coordinate_1=mock_right_ascension,
                                                                        coordinate_2=mock_declination,
                                                                        timestamps=mock_timestamps,
                                                                        n_pointings=5,
                                                                        n_centre_observations=3,
                                                                        distance_threshold=1)

        from matplotlib import pyplot as plt
        colors = ['red'] * 3 + ['blue', 'green', 'yellow', 'orange']
        for i, (d, c) in enumerate(zip(pointing_dumps, pointing_centres)):
            color = colors[i]
            plt.scatter(mock_right_ascension[d], mock_declination[d], color=color)
            plt.scatter(c[0], c[1], color=color, marker='x')

        plt.show()

        for i, (d, c) in enumerate(zip(pointing_dumps[:3], pointing_centres[:3])):
            color = colors[3 + i]
            plt.scatter(mock_right_ascension[d], mock_declination[d], color=color)
            plt.scatter(c[0], c[1], color=color, marker='x')

        plt.show()

    def test_atleast_2d(self):
        pass

    def test_get_ordered_cluster_labels(self):
        pass

    def test_calibrator_pointings_outlier_cluster(self):
        pass

    def test_get_outlier_cluster(self):
        pass

    def test_condense_nested_cluster_list(self):
        pass

    def test_get_cluster_distances(self):
        pass


if __name__ == '__main__':
    unittest.main()
