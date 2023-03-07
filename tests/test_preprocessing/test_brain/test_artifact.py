import unittest

import numpy as np

from brain_pipe.preprocessing.brain.artifact import (
    InterpolateArtifacts,
    ArtifactRemovalMWF,
)


class InterpolateArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = np.reshape(np.arange(24.0) / 24.0, (4, 6))
        self.test_data[0, 4] = 5
        self.test_data[1, 2:5] = [4, 5, 4]
        self.test_data[2, :2] = 5
        self.test_data[3, 4:] = 5

    def test_threshold(self):
        blank_artifacts = InterpolateArtifacts(threshold=3, data_key="data")
        output = blank_artifacts({"data": self.test_data})
        self.assertTrue(
            np.isclose(
                output["data"],
                [
                    [0.0, 0.04166667, 0.08333333, 0.125, 5.0, 0.20833333],
                    [0.25, 0.29166667, 4.0, 4.0, 4.0, 0.45833333],
                    [5, 5.0, 0.58333333, 0.625, 0.66666667, 0.70833333],
                    [0.75, 0.79166667, 0.83333333, 0.875, 5.0, 5.0],
                ],
            ).all()
        )

        blank_artifacts = InterpolateArtifacts(threshold=5, data_key="data")
        output = blank_artifacts({"data": self.test_data})
        self.assertTrue(np.isclose(output["data"], self.test_data).all())


class ArtifactRemovalMWFTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(41)
        self.channels, self.time = 8, 1000
        factor = 0.1
        multiplier = 10

        self.dummy_data = np.random.normal(size=(self.channels, self.time))
        spatial_sinc = np.zeros((self.channels, self.time))
        for midpoint in [
            self.time // 6,
            2 * self.time // 6,
            3 * self.time // 6,
            4 * self.time // 6,
            5 * self.time // 6,
        ]:
            spatial_sinc += self._get_spatial_sinc(midpoint, factor, multiplier)

        self.dummy_data += spatial_sinc
        self.true_mask_indices = [
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            323,
            324,
            325,
            326,
            327,
            328,
            329,
            330,
            331,
            332,
            333,
            334,
            335,
            336,
            337,
            338,
            339,
            340,
            341,
            342,
            343,
            344,
            487,
            488,
            489,
            490,
            491,
            493,
            494,
            495,
            496,
            497,
            498,
            499,
            500,
            501,
            502,
            503,
            504,
            505,
            506,
            507,
            508,
            509,
            510,
            661,
            662,
            663,
            664,
            665,
            666,
            667,
            668,
            669,
            670,
            671,
            672,
            673,
            674,
            675,
            676,
            824,
            825,
            826,
            827,
            828,
            829,
            830,
            831,
            832,
            833,
            834,
            835,
            838,
            839,
            840,
            841,
            842,
        ]
        # self.dummy_data = (self.dummy_data - self.dummy_data.mean(axis=1, keepdims=True))/ self.dummy_data.std(axis=1, keepdims=True)

    def _get_spatial_sinc(self, midpoint, factor, multiplier):
        x = np.arange(self.time)
        sinc = np.sin(factor * (x - midpoint)) / (factor * (x - midpoint))
        sinc[midpoint] = 1
        full_sinc = np.tile(sinc, (self.channels, 1)) * multiplier
        spatial_spread = np.exp(-np.arange(self.channels) / (self.channels / 2))
        spatial_sinc = full_sinc * np.tile(spatial_spread, (self.time, 1)).T
        return spatial_sinc

    def test_apply_mwf(self):
        eye = np.eye(self.channels)
        means = np.mean(self.dummy_data, axis=1)
        full_means = np.tile(means, (self.time, 1)).T
        mwf = ArtifactRemovalMWF()
        output = mwf.apply_mwf(self.dummy_data, eye)
        self.assertTrue(np.isclose(output[0], full_means).all())
        self.assertTrue(np.isclose(output[1], self.dummy_data - full_means).all())

    def test_check_symmetric(self):
        mat1 = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        mat2 = np.array([[1, 7, 3], [7, 4, 5], [3, 5, 1]])
        mat3 = np.array([[1, 2, 3], [2, 4, 1], [3, 5, 6]])
        mat4 = np.array([[1, 6.99, 3], [7, 4, 5], [3, 5, 1]])

        mwf = ArtifactRemovalMWF()
        self.assertTrue(mwf.check_symmetric(mat1))
        self.assertTrue(mwf.check_symmetric(mat2))
        self.assertFalse(mwf.check_symmetric(mat3))
        self.assertFalse(mwf.check_symmetric(mat4))

    def test_compute_mwf(self):
        mwf = ArtifactRemovalMWF()
        mask = np.zeros(self.time)
        # hardcoded output from mwf.get_artifact_segments
        mask[self.true_mask_indices] = 1
        output = mwf.compute_mwf(self.dummy_data, mask.astype(bool))
        self.assertEqual(output.shape, (self.channels * 7, self.channels * 7))
        self.assertTrue(np.isclose(output.max(), (0.3008365160662489 + 0j)))
        self.assertTrue(np.isclose(output.min(), (-0.13151094589088408 + 0j)))
        self.assertTrue(np.isclose(output.mean(), (0.001943452199005853 + 0j)))
        self.assertTrue(np.isclose(output.std(), 0.042316874554468635))

    def test_fix_symmetric(self):
        mat1 = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        mat2 = np.array([[1, 6.99, 3], [7, 4, 5], [3, 5, 1]])
        mat2_sym = np.array([[1, 6.995, 3], [6.995, 4, 5], [3, 5, 1]])

        mwf = ArtifactRemovalMWF()
        self.assertTrue((mwf.fix_symmetric(mat1) == mat1).all())
        self.assertTrue((mwf.fix_symmetric(mat2) == mat2_sym).all())

    def test_get_artifact_segments(self):
        mwf = ArtifactRemovalMWF(reference_channels=[0, 1, 2, 3])
        output = mwf.get_artifact_segments(self.dummy_data.T, 5)
        self.assertEqual(np.where(output)[0].tolist(), self.true_mask_indices)

    def test_sort_evd(self):
        eig_vals = np.array([3, 1, 2])
        eig_vecs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output = ArtifactRemovalMWF().sort_evd(eig_vals, eig_vecs)
        self.assertEqual(output[0].tolist(), [3, 2, 1])
        self.assertEqual(output[1].tolist(), [[1, 3, 2], [4, 6, 5], [7, 9, 8]])

    def test_stack_delayed(self):
        mwf = ArtifactRemovalMWF()
        output, nb_channels = mwf.stack_delayed(self.dummy_data, 5)
        self.assertEqual(output.shape, (nb_channels, self.time))
        self.assertTrue(
            (
                output[: self.channels, :-1]
                == output[self.channels : 2 * self.channels, 1:]
            ).all()
        )

    def test_call(self):
        np.random.seed(42)
        mwf = ArtifactRemovalMWF(reference_channels=[0, 1, 2, 3])
        output = mwf({"data": self.dummy_data, "data_fs": 5})
        self.assertEqual(output["data"].shape, (self.channels, self.time))
        self.assertTrue(np.isclose(output["data"].max(), 10.958065251240939))
        self.assertTrue(np.isclose(output["data"].min(), -4.80211082577799))
        self.assertTrue(np.isclose(output["data"].mean(), 0.7806721594497221))
        self.assertTrue(np.isclose(output["data"].std(), 2.188922034921533))
