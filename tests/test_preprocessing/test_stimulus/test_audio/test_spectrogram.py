import os
import unittest

import librosa
import numpy as np

from brain_pipe.preprocessing.stimulus.audio.spectrogram import LibrosaMelSpectrogram
from examples.exporl.sparrKULee import SparrKULeeSpectrogramKwargs

test_data_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "test_data",
)
test_wav_path = os.path.join(test_data_path, "OSR_us_000_0010_8k.wav")


class LibrosaMelSpectrogramTest(unittest.TestCase):
    def test_mel_spectrogram_default(self):
        data, sr = librosa.load(test_wav_path, sr=None)
        output = LibrosaMelSpectrogram()({"stimulus_data": data, "stimulus_sr": sr})
        self.assertEqual(output["spectrogram_data"].shape, (526, 128))
        self.assertEqual(output["spectrogram_sr"], 15.625)
        self.assertTrue(
            np.isclose(
                [
                    9.93085384e-01,
                    6.33554101e-01,
                    4.33867723e-02,
                    9.22227725e-02,
                    7.99714699e-02,
                    4.78277989e-02,
                    4.07961570e-02,
                    7.16419592e-02,
                    4.77283478e-01,
                    6.53233719e00,
                    1.34911995e01,
                    1.61980839e01,
                    7.15204859e00,
                    3.83730030e00,
                    1.05433607e00,
                    2.18878686e-01,
                    8.85021538e-02,
                    6.47132546e-02,
                    1.87550575e-01,
                    1.22151792e00,
                    3.11051726e00,
                    3.84796500e00,
                    3.48256969e00,
                    3.18433928e00,
                    1.60637641e00,
                    1.75314832e00,
                    1.05678082e00,
                    7.15241492e-01,
                    6.18904591e-01,
                    4.12648946e-01,
                    4.72674817e-01,
                    3.33862871e-01,
                    1.38370693e-01,
                    1.04700223e-01,
                    9.21346694e-02,
                    1.11439511e-01,
                    4.50547077e-02,
                    1.89716127e-02,
                    1.69036202e-02,
                    1.84712000e-02,
                    5.28618060e-02,
                    1.43894002e-01,
                    1.88596264e-01,
                    2.18813896e-01,
                    3.64188820e-01,
                    3.50929707e-01,
                    4.92889345e-01,
                    5.66330492e-01,
                    3.60140413e-01,
                    1.83673322e-01,
                    1.35173693e-01,
                    3.72063220e-01,
                    3.42203259e-01,
                    1.60684034e-01,
                    1.56894729e-01,
                    2.17115119e-01,
                    2.42398605e-01,
                    2.69342124e-01,
                    2.18408570e-01,
                    1.11492418e-01,
                    5.90992570e-02,
                    1.00334600e-01,
                    5.97917363e-02,
                    3.82230617e-02,
                    3.49829122e-02,
                    5.55102676e-02,
                    5.63279055e-02,
                    8.41718763e-02,
                    8.70513842e-02,
                    5.29253185e-02,
                    3.88142057e-02,
                    2.13917494e-02,
                    1.62857193e-02,
                    1.97865497e-02,
                    2.75023766e-02,
                    3.72895859e-02,
                    3.44561376e-02,
                    2.26456486e-02,
                    2.47140247e-02,
                    1.92838032e-02,
                    1.88388601e-02,
                    1.48547944e-02,
                    2.02053320e-02,
                    1.34915458e-02,
                    9.32857580e-03,
                    9.15729720e-03,
                    8.95753875e-03,
                    1.04684290e-02,
                    6.29476178e-03,
                    5.23403287e-03,
                    3.78717552e-03,
                    3.27364844e-03,
                    3.43172648e-03,
                    4.02672542e-03,
                    4.01608180e-03,
                    3.33369779e-03,
                    2.89192353e-03,
                    2.67943996e-03,
                    2.18655658e-03,
                    1.71516801e-03,
                    9.84668382e-04,
                    6.55685202e-04,
                    8.12099257e-04,
                    1.02274551e-03,
                    1.97318825e-03,
                    2.66572880e-03,
                    2.55632354e-03,
                    2.02497491e-03,
                    2.39151670e-03,
                    3.11306072e-03,
                    3.43276025e-03,
                    3.23792011e-03,
                    4.17931005e-03,
                    4.65961359e-03,
                    5.41556487e-03,
                    4.60625207e-03,
                    3.79136647e-03,
                    2.98407790e-03,
                    1.79715757e-03,
                    1.32729451e-03,
                    1.12456898e-03,
                    1.01733115e-03,
                    1.25385483e-03,
                    2.06709164e-03,
                    3.94276716e-03,
                    7.64125632e-03,
                    9.95123573e-03,
                    1.04924822e-02,
                ],
                output["spectrogram_data"].mean(axis=0),
            ).all()
        )

    def test_mel_spectrogram_callable_item(self):
        data, sr = librosa.load(test_wav_path, sr=None)
        output = LibrosaMelSpectrogram(
            librosa_kwargs={"hop_length": lambda x, y: 2048}
        )({"stimulus_data": data, "stimulus_sr": sr})
        self.assertEqual(output["spectrogram_data"].shape, (132, 128))
        self.assertEqual(output["spectrogram_sr"], 3.90625)

    def test_mel_spectrogram_callable(self):
        data, sr = librosa.load(test_wav_path, sr=None)

        def test_fn(data_dict):
            return {"hop_length": 2048}

        output = LibrosaMelSpectrogram(librosa_kwargs=test_fn)(
            {"stimulus_data": data, "stimulus_sr": sr}
        )
        self.assertEqual(output["spectrogram_data"].shape, (132, 128))
        self.assertEqual(output["spectrogram_sr"], 3.90625)
