import unittest

import numpy as np
import scipy.signal

from brain_pipe.preprocessing.filter import SosFiltFilt


class SosFiltFiltTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.test_data = np.random.normal(size=(4, 4))
        self.matlab_filtered_data_1 = [
            [
                0.5787777268587571,
                0.7115014361825422,
                0.8636020906207311,
                1.023287126275008,
            ],
            [
                -0.2868883588731042,
                -0.1279355530296372,
                0.017415819508094503,
                0.12728701556954478,
            ],
            [
                -0.5285998571314571,
                -0.5179524997984798,
                -0.5290908826997855,
                -0.5526339798272134,
            ],
            [
                0.43954159443969565,
                0.28334311273488744,
                0.1902564550028823,
                0.16266724400115273,
            ],
        ]
        self.matlab_filtered_data_2 = [
            [
                0.5820267506009155,
                -0.2652950907362652,
                0.5334707908358344,
                1.573134005072963,
            ],
            [
                0.5272786339459519,
                -0.49356091277130937,
                0.21066830778727935,
                1.2596294461842923,
            ],
            [
                0.4975110220820324,
                -0.740538891421736,
                -0.13662335072141735,
                0.9700882848361521,
            ],
            [
                0.4963580988089613,
                -1.006127083818106,
                -0.4787620742316074,
                0.7256720792763162,
            ],
        ]
        self.test_data_2 = np.random.normal(size=(7, 2))

        self.filtered_data_0 = np.array(
            [
                [-1.0475693618067052, 0.3301816493557466],
                [-0.9289497565990208, 0.19849960042269407],
                [-0.8333749224655532, 0.101397265899585],
                [-0.7909342613581236, 0.04048169654234372],
                [-0.789831526899412, 0.010189819560548345],
                [-0.7983979623962235, -0.011599844787299222],
                [-0.8044820343688608, -0.0394336190416007],
            ]
        )
        self.filter_ = scipy.signal.butter(1, 0.5, "lowpass", output="sos", fs=16)

    def test_matlab_mode(self):
        """Test MATLAB mode."""
        filtfilt = SosFiltFilt(self.filter_, emulate_matlab=True)
        output = filtfilt({"data": self.test_data})
        self.assertTrue(np.isclose(output["data"], self.matlab_filtered_data_1).all())

        # Try to adapt padding
        filtfilt = SosFiltFilt(
            self.filter_, emulate_matlab=True, padtype="even", padlen=1
        )
        output = filtfilt({"data": self.test_data})
        self.assertTrue(np.isclose(output["data"], self.matlab_filtered_data_1).all())

        # Only axis should make a difference
        filtfilt = SosFiltFilt(
            self.filter_,
            emulate_matlab=True,
            axis=0,
        )
        output = filtfilt({"data": self.test_data})
        self.assertTrue(np.isclose(output["data"], self.matlab_filtered_data_2).all())

        filtfilt = SosFiltFilt(
            self.filter_,
            emulate_matlab=True,
            axis=0,
            padtype="even",
            padlen=1,
        )
        output = filtfilt({"data": self.test_data})
        self.assertTrue(np.isclose(output["data"], self.matlab_filtered_data_2).all())

    def test_callable(self):
        """Test callable filter."""
        filtfilt = SosFiltFilt(
            lambda d: scipy.signal.butter(1, 0.5, "lowpass", output="sos", fs=d["fs"]),
            emulate_matlab=False,
            axis=0,
        )
        output = filtfilt({"data": self.test_data_2, "fs": 16})
        self.assertTrue(np.isclose(output["data"], self.filtered_data_0).all())

    def test_matlab_mode_callable(self):
        """Test with callable filter + emulate_matlab."""
        filtfilt = SosFiltFilt(
            lambda d: scipy.signal.butter(1, 0.5, "lowpass", output="sos", fs=d["fs"]),
            emulate_matlab=True,
            axis=0,
            padtype="even",
            padlen=1,
        )
        output = filtfilt({"data": self.test_data, "fs": 16})
        self.assertTrue(np.isclose(output["data"], self.matlab_filtered_data_2).all())

    def test_multiple_keys(self):
        """Test multiple keys."""
        # Test with multiple keys
        filtfilt = SosFiltFilt(
            self.filter_, ["data", "data2"], emulate_matlab=False, axis=0
        )
        output = filtfilt({"data": self.test_data_2, "data2": self.test_data_2[:, 1]})
        self.assertTrue(np.isclose(output["data"], self.filtered_data_0).all())
        self.assertTrue(np.isclose(output["data2"], self.filtered_data_0[:, 1]).all())

        # Ignore one key
        filtfilt = SosFiltFilt(self.filter_, emulate_matlab=False, axis=0)
        output = filtfilt({"data": self.test_data_2, "data2": self.test_data_2[:, 1]})
        self.assertTrue(np.isclose(output["data"], self.filtered_data_0).all())
        self.assertTrue(np.isclose(output["data2"], self.test_data_2[:, 1]).all())

        # Test mapping
        filtfilt = SosFiltFilt(
            self.filter_, {"data": "data2"}, emulate_matlab=False, axis=0
        )
        output = filtfilt({"data": self.test_data_2[:, 1]})
        self.assertTrue(np.isclose(output["data"], self.test_data_2[:, 1]).all())
        self.assertTrue(np.isclose(output["data2"], self.filtered_data_0[:, 1]).all())
