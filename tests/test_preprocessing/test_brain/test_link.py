import os
import tempfile
import unittest

from brain_pipe.preprocessing.brain.link import (
    BIDSStimulusInfoExtractor,
    BasenameComparisonFn,
    LinkStimulusToBrainResponse,
)


class BIDSStimulusInfoExtractorTest(unittest.TestCase):
    def test_info_extract(self):
        events = [
            "onset\tduration\tstim_file\ttrial_type",
            "1.0\t32.0\tstimuli_1.jpg\ttype_a",
            "3.0\t4.0\tstimuli_1.wav\ttype_b",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "sub-01_ses-01_task-01_run-01_bold.nii.gz")
            event_path = os.path.join(tmpdir, "sub-01_ses-01_task-01_run-01_events.tsv")
            with open(event_path, "w") as fp:
                fp.write("\n".join(events))
            extractor = BIDSStimulusInfoExtractor()
            output = extractor({"data_path": data_path})
        self.assertEqual(
            output,
            [
                {
                    "onset": "1.0",
                    "duration": "32.0",
                    "stim_file": "stimuli_1.jpg",
                    "trial_type": "type_a",
                },
                {
                    "onset": "3.0",
                    "duration": "4.0",
                    "stim_file": "stimuli_1.wav",
                    "trial_type": "type_b",
                },
            ],
        )


class BasenameComparisonFnTest(unittest.TestCase):
    def test_no_ignore_extension(self):
        comparison = BasenameComparisonFn()
        output = comparison(
            [{"a": "stimulus.wav"}, {"b": "stimulus.jpg"}],
            {"stimulus_path": "stimulus.wav"},
        )
        self.assertTrue(output)
        output = comparison([{"b": "stimulus.jpg"}], {"stimulus_path": "stimulus.wav"})
        self.assertFalse(output)

    def test_ignore_extension(self):
        comparison = BasenameComparisonFn(ignore_extension=True)
        output = comparison(
            [{"a": "stimulus.wav"}, {"b": "stimulus.jpg"}],
            {"stimulus_path": "stimulus.wav"},
        )
        self.assertTrue(output)
        output = comparison([{"b": "stimulus.jpg"}], {"stimulus_path": "stimulus.wav"})
        self.assertTrue(output)
        output = comparison([{"b": "something.wav"}], {"stimulus_path": "stimulus.wav"})
        self.assertFalse(output)

    def test_process_path(self):
        comparison = BasenameComparisonFn()
        self.assertEqual(comparison.process_path("a"), "a")
        self.assertEqual(comparison.process_path(1234), None)
        comparison = BasenameComparisonFn(ignore_extension=True)
        self.assertEqual(comparison.process_path("/a/b/c.d.e.f"), "c.d.e")


class LinkStimulusToBrainResponseTest(unittest.TestCase):
    class StimExtractorMockup:
        def __call__(self, brain_dict):
            return [{"stim_path": "stimulus.wav"}, {"a": "other1.wav"}]

    class GroupFnMockup:
        def __call__(self, events_row):
            events_row["stimulus_path"] = "a"
            return events_row

    def test_dicts(self):
        linker = LinkStimulusToBrainResponse(
            stimulus_data=[
                {"stimulus_path": "stimulus.wav"},
                {"stimulus_path": "other2.wav"},
            ],
            extract_stimuli_information_fn=self.StimExtractorMockup(),
        )
        self.assertEqual(linker({}), {"stimuli": [{"stimulus_path": "stimulus.wav"}]})

    def test_callable(self):
        def mockup_callable(data_dict):
            data_dict["c"] = "c"
            return data_dict

        with self.assertRaises(ValueError):
            LinkStimulusToBrainResponse(
                stimulus_data=mockup_callable,
                extract_stimuli_information_fn=self.StimExtractorMockup(),
            )
        linker = LinkStimulusToBrainResponse(
            stimulus_data=mockup_callable,
            extract_stimuli_information_fn=self.StimExtractorMockup(),
            grouper=self.GroupFnMockup(),
        )
        self.assertEqual(
            linker({}),
            {
                "stimuli": [
                    {"c": "c", "stim_path": "stimulus.wav", "stimulus_path": "a"},
                    {"a": "other1.wav", "c": "c", "stimulus_path": "a"},
                ]
            },
        )
