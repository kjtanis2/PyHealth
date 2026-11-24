"""SleepApneaDetection, on small synthetic EDF and annotation files.

The task is called on stand-in patients, so no dataset is loaded here; the
NCH and CHAT test files run it through the datasets.
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd

from pyhealth.tasks import SleepApneaDetection
from pyhealth.tasks.sleep_apnea_detection import (
    _event_seconds,
    detect_r_peaks,
)
from tests.core.psg_test_utils import write_edf, write_tsv

SFREQ = 8
SECONDS = 120
NCH_CHANNELS = ("EOG LOC-M2", "EEG C4-M1", "Resp Airflow", "SpO2", "Capno")

# Wake, then three sleep epochs with an apnea, a hypopnea and an apnea.
ROWS = [
    (0, 30, "Sleep stage W"),
    (30, 30, "Sleep stage N2"),
    (60, 30, "Sleep stage N3"),
    (90, 30, "Sleep stage R"),
    (35.5, 12.0, "Obstructive Apnea"),
    (65.0, 15.0, "Hypopnea"),
    (95.0, 10.0, "Central Apnea"),
    (66.0, 3.0, "EEG arousal"),
]


def _patient(*studies) -> SimpleNamespace:
    """A stand-in patient whose ``studies`` events are the given dicts."""
    events = [SimpleNamespace(**study) for study in studies]
    return SimpleNamespace(patient_id="p1", get_events=lambda table: events)


class TestSleepApneaDetection(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _study(
        self,
        name="s1",
        labels=NCH_CHANNELS,
        rows=ROWS,
        sfreq=SFREQ,
        seconds=SECONDS,
        **edf,
    ) -> dict:
        signal_file = self.root / f"{name}.edf"
        annotation_file = self.root / f"{name}.tsv"
        write_edf(signal_file, labels, sfreq=sfreq, seconds=seconds, **edf)
        write_tsv(annotation_file, rows)
        return {
            "record_id": name,
            "signal_file": str(signal_file),
            "annotation_file": str(annotation_file),
            "ahi": "10.0",
        }

    def test_windows_labels_and_wake(self):
        task = SleepApneaDetection(
            modalities=["eog", "eeg", "resp", "spo2", "co2"], sampling_rate=SFREQ
        )

        samples = task(_patient(self._study()))

        # The first window is wake and is dropped.
        self.assertEqual([s["start_second"] for s in samples], [30, 60, 90])
        self.assertEqual([s["apnea_seconds"] for s in samples], [12, 0, 10])
        self.assertEqual([s["hypopnea_seconds"] for s in samples], [0, 15, 0])
        self.assertEqual([s["label"] for s in samples], [1, 1, 1])
        self.assertEqual(samples[0]["signal"].shape, (5, 30 * SFREQ))
        self.assertEqual(samples[0]["signal"].dtype, np.float32)
        self.assertEqual({s["patient_id"] for s in samples}, {"p1"})

    def test_window_without_events_is_negative(self):
        rows = [(0, 60, "Sleep stage N2"), (60, 60, "Sleep stage R")]
        task = SleepApneaDetection(modalities=["spo2"], sampling_rate=SFREQ)

        samples = task(_patient(self._study(rows=rows)))

        self.assertEqual([s["label"] for s in samples], [0, 0, 0, 0])

    def test_event_seconds_match_mne_chunking(self):
        rows = pd.DataFrame(
            [
                (35.5, 12.0, "Apnea"),
                (10.2, 1.0, "Apnea"),
                (50.0, 0.5, "Apnea"),
                (60.0, 2.7, "Apnea"),
            ],
            columns=["onset", "duration", "description"],
        )
        raw = mne.io.RawArray(
            np.zeros((1, 100 * 64)), mne.create_info(1, 64.0), verbose="error"
        )
        raw.set_annotations(mne.Annotations(rows.onset, rows.duration, "Apnea"))
        events, _ = mne.events_from_annotations(
            raw, event_id={"Apnea": 1}, chunk_duration=1.0, verbose="error"
        )

        seconds = _event_seconds(rows, frozenset({"Apnea"}))

        self.assertEqual(seconds, set((events[:, 0] / 64).astype(int).tolist()))
        self.assertEqual(len(seconds), 12 + 1 + 0 + 2)

    def test_channel_order_and_units(self):
        n = SECONDS * SFREQ
        study = self._study(
            signals={"EEG C4-M1": np.full(n, 50.0), "SpO2": np.full(n, 97.0)},
            units={"SpO2": "%"},
        )
        task = SleepApneaDetection(modalities=["spo2", "eeg"], sampling_rate=SFREQ)

        samples = task(_patient(study))

        self.assertEqual(task.channel_names, ["eeg", "spo2"])
        np.testing.assert_allclose(samples[0]["signal"][0], 50.0, atol=0.01)
        np.testing.assert_allclose(samples[0]["signal"][1], 97.0, atol=0.01)

    def test_chat_electrodes_are_rereferenced(self):
        n = SECONDS * SFREQ
        study = self._study(
            labels=("C4", "M1", "SaO2"),
            signals={"C4": np.full(n, 60.0), "M1": np.full(n, 10.0)},
            rows=[(0, 120, "Stage 2 sleep"), (40.0, 11.0, "Hypopnea")],
        )
        task = SleepApneaDetection(modalities=["eeg", "spo2"], sampling_rate=SFREQ)

        samples = task(_patient(study))

        self.assertEqual([s["label"] for s in samples], [0, 1, 0, 0])
        np.testing.assert_allclose(samples[0]["signal"][0], 50.0, atol=0.01)

    def test_resampling_to_sampling_rate(self):
        task = SleepApneaDetection(modalities=["eeg"], sampling_rate=2 * SFREQ)

        samples = task(_patient(self._study()))

        self.assertEqual(samples[0]["signal"].shape, (1, 30 * 2 * SFREQ))

    def test_unreachable_sampling_rate_raises(self):
        task = SleepApneaDetection(modalities=["eeg"], sampling_rate=math.pi)

        with self.assertRaises(ValueError):
            task(_patient(self._study()))

    def test_missing_modality_skips_the_study_or_is_filled(self):
        study = self._study(labels=("EEG C4-M1",))

        with self.assertLogs("pyhealth.tasks.sleep_apnea_detection", "WARNING"):
            skipped = SleepApneaDetection(
                modalities=["eeg", "co2"], sampling_rate=SFREQ
            )(_patient(study))
        filled = SleepApneaDetection(
            modalities=["eeg", "co2"], sampling_rate=SFREQ, fill_missing=True
        )(_patient(study))

        self.assertEqual(skipped, [])
        self.assertEqual(len(filled), 3)
        self.assertFalse(filled[0]["signal"][1].any())

    def test_min_ahi_leaves_out_studies(self):
        low = {**self._study("low"), "ahi": "2.5"}
        unknown = {**self._study("unknown"), "ahi": ""}
        high = {**self._study("high"), "ahi": "3.0"}
        task = SleepApneaDetection(modalities=["eeg"], sampling_rate=SFREQ, min_ahi=3.0)

        samples = task(_patient(low, unknown, high))

        self.assertEqual({s["record_id"] for s in samples}, {"high"})

    def test_unreadable_or_short_studies_are_skipped(self):
        unreadable = {**self._study("bad"), "annotation_file": "missing.tsv"}
        short = self._study("short", seconds=20)
        task = SleepApneaDetection(modalities=["eeg"], sampling_rate=SFREQ)

        with self.assertLogs("pyhealth.tasks.sleep_apnea_detection", "WARNING") as logs:
            samples = task(_patient(unreadable, short))

        self.assertEqual(samples, [])
        self.assertEqual(len(logs.output), 2)

    def test_ecg_becomes_rr_interval_and_r_amplitude(self):
        sfreq, seconds = 128, 60
        ecg = np.zeros(sfreq * seconds)
        ecg[: 30 * sfreq : 96] = 800.0  # one beat every 0.75 s, first window
        study = self._study(
            labels=("ECG EKG2-EKG",),
            sfreq=sfreq,
            seconds=seconds,
            signals={"ECG EKG2-EKG": ecg},
            rows=[(0, 60, "Sleep stage N2")],
        )
        task = SleepApneaDetection(modalities=["ecg"], sampling_rate=64)

        first, second = task(_patient(study))

        self.assertEqual(task.channel_names, ["ecg_rri", "ecg_r_amplitude"])
        self.assertEqual(first["signal"].shape, (2, 30 * 64))
        middle = slice(5 * 64, 25 * 64)
        np.testing.assert_allclose(first["signal"][0, middle], 0.75, atol=0.01)
        self.assertGreater(first["signal"][1, middle].mean(), 100.0)
        # No beats: R-R interval and amplitude are zeros, as in the original.
        self.assertFalse(second["signal"].any())

    def test_detect_r_peaks_finds_regular_beats(self):
        ecg = np.zeros(1280)
        ecg[64::128] = 1.0

        peaks = detect_r_peaks(ecg, 128.0)

        np.testing.assert_array_equal(peaks, np.arange(64, 1280, 128))
        self.assertEqual(len(detect_r_peaks(np.zeros(1280), 128.0)), 0)

    def test_invalid_arguments_raise(self):
        for kwargs in (
            {"modalities": ["eeg", "emg"]},
            {"modalities": []},
            {"modalities": ["eeg", "eeg"]},
            {"sampling_rate": 0},
            {"sampling_rate": math.nan},
            {"window_seconds": 2.5},
            {"window_seconds": True},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SleepApneaDetection(**kwargs)


if __name__ == "__main__":
    unittest.main()
