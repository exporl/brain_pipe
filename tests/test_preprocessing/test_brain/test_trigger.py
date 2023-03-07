import unittest

import numpy as np

from brain_pipe.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
    default_drift_correction,
)
from brain_pipe.preprocessing.brain.eeg.biosemi import (
    biosemi_trigger_processing_fn,
)


class DefaultDriftCorrectionTest(unittest.TestCase):
    def test_default_drift_correction(self):
        np.random.seed(42)
        brain_indices = np.arange(1, 109, 10)
        brain_trigger_indices = brain_indices + np.random.randint(
            -1, 2, size=brain_indices.shape
        )
        brain_data = np.random.normal(0, 1, size=(10, 110))
        brain_fs = 4
        stimulus_trigger_indices = np.arange(0, 1001, 100)
        stimulus_fs = 8

        output = default_drift_correction(
            brain_data,
            brain_trigger_indices,
            brain_fs,
            stimulus_trigger_indices,
            stimulus_fs,
        )
        self.assertEqual(output.shape, (10, 107))
        self.assertTrue(np.isclose(output[:, 0], brain_data[:, 2]).all())
        self.assertTrue(
            np.isclose(
                np.mean(output, axis=1),
                [
                    0.013305166514305168,
                    0.2109117127832399,
                    -0.1939594468025794,
                    0.09905934958959388,
                    -0.024950108211425164,
                    0.05937828078922733,
                    -0.2785785626309968,
                    0.16793931966957623,
                    0.11640091744500194,
                    0.42827322756444086,
                ],
            ).all()
        )
        self.assertTrue(
            np.isclose(
                np.std(output, axis=1),
                [
                    0.6013185253331836,
                    0.8147063017795653,
                    0.6986282178144578,
                    0.8099040065378291,
                    0.880285216468808,
                    0.8193050329687783,
                    0.9091001676748534,
                    0.9235396362593349,
                    0.6828021507245129,
                    0.8734865633522776,
                ],
            ).all()
        )


class BioSemiTriggerProcessingFnTest(unittest.TestCase):
    def test_biosemi_trigger_processing_fn(self):
        np.random.seed(42)
        triggers = np.array([253] * 1000)
        triggers[[100, 200, 300, 400, 500]] = 254
        triggers[[101, 702]] = 253 + 2**16
        trigger_mask = biosemi_trigger_processing_fn(triggers)
        self.assertEqual(np.where(trigger_mask)[0].tolist(), [100, 200, 300, 400, 500])


class AlignPeriodicBlockTriggersTest(unittest.TestCase):
    def test_get_trigger_indices(self):
        aligner = AlignPeriodicBlockTriggers()
        dummy_trigger = (
            [0.0] * 6
            + [1.0] * 2
            + [0.0] * 3
            + [1.0] * 5
            + [0.0] * 2
            + [1.0] * 2
            + [0.0]
        )
        indices = aligner.get_trigger_indices(np.array(dummy_trigger))
        self.assertTrue((indices == np.array([6, 11, 18])).all())

        dummy_trigger = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        indices = aligner.get_trigger_indices(np.array(dummy_trigger))
        self.assertTrue((indices == np.array([0, 2, 4])).all())
