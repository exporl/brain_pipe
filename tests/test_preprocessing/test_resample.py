import unittest
from collections import OrderedDict

import numpy as np

from brain_pipe.preprocessing.resample import ResamplePoly


class ResamplePolyTest(unittest.TestCase):
    def setUp(self):
        """Create a simple 2D array of data to test on."""
        self.data = np.sin(np.reshape(np.arange(64), (-1, 2)))

        self.downsampled_0 = [
            [0.24804234832093033, 0.34493009688114623],
            [-0.06384682386366039, -0.12529857816606493],
            [0.031623030559972135, 0.07112201726026349],
            [-0.018995635020249195, -0.04480587713405009],
            [0.012257777841964409, 0.029129534521437385],
            [-0.00822867189900121, -0.01906315373393308],
            [0.005803980149567839, 0.012349034963862422],
            [-0.004327304329220834, -0.008544780992047332],
            [0.004026480397434071, 0.007301288894505034],
            [-0.005090468663122039, -0.008532635156254955],
            [0.007798886223035088, 0.012317285836560242],
            [-0.012757297730959849, -0.018991080206284334],
            [0.019804423770505557, 0.029009428905701146],
            [-0.030323609807566702, -0.04462559129584356],
            [0.0466888102839532, 0.07088224388710825],
            [-0.07425992164335127, -0.12513285268584576],
        ]

    def test_axis(self):
        """Test that the axis parameter works as expected."""
        downsampled = [
            [0.2664941517686039],
            [0.49957673608853537],
            [-0.6822887084401852],
            [0.06828783916990186],
            [0.6254531719497958],
            [-0.5888485570004687],
            [-0.13535824354760634],
            [0.7015063667062968],
            [-0.4485010670974036],
            [-0.32822176618509236],
            [0.7216779664650876],
            [-0.27242623931534954],
            [-0.4949393310980529],
            [0.6843611131537762],
            [-0.07465009349159707],
            [-0.6222303126448232],
            [0.5925284459133622],
            [0.12907263598274277],
            [-0.6999547842114007],
            [0.4534953023684814],
            [0.322513513272134],
            [-0.7219212589522881],
            [0.2783369830261162],
            [0.4902631489915003],
            [-0.6863799000830226],
            [0.08100649918668679],
            [0.6189587033304259],
            [-0.5961619118752426],
            [-0.12277691593686887],
            [0.6983483622115274],
            [-0.4584540075475413],
            [-0.3167799923249893],
        ]
        output = ResamplePoly(16, "data", "fs", axis=1)({"data": self.data, "fs": 32})
        self.assertEqual(output["fs"], 16)
        self.assertTrue(np.isclose(output["data"], downsampled).all())

        output = ResamplePoly(16, "data", "fs", axis=0)({"data": self.data, "fs": 32})
        self.assertEqual(output["fs"], 16)
        self.assertTrue(np.isclose(output["data"], self.downsampled_0).all())

    def test_keys(self):
        """Test the keys of the output dict."""

        downsampled2 = [
            [
                0.189253676373305,
                0.2592081023737745,
                -0.40499093999377317,
                0.07786329464355007,
                0.34018581249567037,
                -0.3609977940601352,
                -0.03972963249842626,
                0.39406451582293645,
                -0.2882477708119664,
                -0.15415771989260535,
                0.4165522657372228,
                -0.1925360953935744,
                -0.2563056916988732,
                0.4058577009726009,
                -0.08148710499721472,
                -0.33803649904464944,
                0.3628327444270187,
                0.03605310146660619,
                -0.3928395126731013,
                0.2909047394726581,
                0.1507213385368667,
                -0.416349155937194,
                0.1958034297478057,
                0.2533832001879341,
                -0.40669266413260585,
                0.08510453106349268,
                0.33586070137680485,
                -0.36463926786041634,
                -0.03237374577484815,
                0.3915837316431853,
                -0.2935389165584252,
                -0.14727314858465676,
            ]
        ]
        downsample = ResamplePoly(16, ["data", "d2"], ["fs", "sr"])
        output = downsample({"data": self.data, "fs": 32, "d2": self.data.T, "sr": 64})
        self.assertEqual(output["fs"], 16)
        self.assertEqual(output["sr"], 16)
        self.assertTrue(np.isclose(output["data"], self.downsampled_0).all())
        self.assertTrue(np.isclose(output["d2"], downsampled2).all())

        # Test mapping
        # For resample an ordered dict is required
        with self.assertRaises(TypeError):
            ResamplePoly(16, {"data": "_data", "d2": "_d2"}, {"fs": "_fs", "sr": "_sr"})
        downsample = ResamplePoly(
            16,
            OrderedDict([("data", "_data"), ("d2", "_d2")]),
            OrderedDict([("fs", "_fs"), ("sr", "_sr")]),
        )
        output = downsample({"data": self.data, "fs": 32, "d2": self.data.T, "sr": 64})
        self.assertEqual(output["_fs"], 16)
        self.assertEqual(output["_sr"], 16)
        self.assertEqual(output["fs"], 32)
        self.assertEqual(output["sr"], 64)
        self.assertTrue(np.isclose(output["data"], self.data).all())
        self.assertTrue(np.isclose(output["d2"], self.data.T).all())
        self.assertTrue(np.isclose(output["_data"], self.downsampled_0).all())
        self.assertTrue(np.isclose(output["_d2"], downsampled2).all())
