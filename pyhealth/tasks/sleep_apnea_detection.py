"""Sleep apnea detection on pediatric polysomnography (NCH and CHAT).

Each sleep study is cut into 30 second windows. A window is labelled 1 when
an apnea or a hypopnea was scored during it, and windows scored as wake are
left out. Every window holds one channel per modality (EOG, EEG, airflow,
SpO2, CO2) plus the R-R interval and R-peak amplitude derived from the ECG,
so models can be trained and tested on any subset of the modalities.

This follows the data pipeline of Fayyaz et al., who study apnea detection
when modalities are missing or noisy (``data/nch/preprocessing.py``,
``data/chat/preprocessing.py`` and the ``dataloader.py`` files of their
code).

References:
    - Fayyaz, H. et al. Multimodal Sleep Apnea Detection with Missing or
      Noisy Modalities. Code: https://github.com/healthylaife/apnea-missing-modality
    - Lee, H. et al. A large collection of real-world pediatric sleep
      studies. Scientific Data 9, 421, 2022.
      https://doi.org/10.1038/s41597-022-01545-6
    - Marcus, C. L. et al. A randomized trial of adenotonsillectomy for
      childhood sleep apnea. NEJM 368, 2366-2376, 2013.
      https://doi.org/10.1056/NEJMoa1215881
"""

import logging
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, ClassVar

import mne
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.interpolate import splev, splrep

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

APNEA_MODALITIES: tuple[str, ...] = ("eog", "eeg", "resp", "spo2", "co2", "ecg")
"""Modalities of the task, in the order of the output channels."""

# Channels of each modality, tried in order: NCH bipolar labels first, then
# the CHAT electrodes. A pair is (channel, reference); the second is
# subtracted from the first. Matching ignores case.
_MONTAGES: dict[str, tuple[tuple[str, ...], ...]] = {
    "eog": (("EOG LOC-M2",), ("E1", "M2")),
    "eeg": (("EEG C4-M1",), ("C4", "M1")),
    "resp": (("RESP AIRFLOW",), ("AIRFLOW",)),
    "spo2": (("SPO2",), ("SAO2",)),
    "co2": (("CAPNO",), ("CAP",)),
    "ecg": (("ECG EKG2-EKG",), ("ECG1", "ECG3")),
}

# The ECG modality becomes two channels.
_OUTPUT_CHANNELS: dict[str, tuple[str, ...]] = {
    **{m: (m,) for m in APNEA_MODALITIES if m != "ecg"},
    "ecg": ("ecg_rri", "ecg_r_amplitude"),
}

# R-R intervals are clipped to this range in seconds, as in the original.
_RRI_RANGE = (0.0, 2.0)

# Largest numerator or denominator of the resampling ratio.
_MAX_RESAMPLING_TERM = 1000

# EDF physical dimensions that mne converts to volts.
_VOLTAGE_UNITS = frozenset({"v", "mv", "uv", "µv", "nv"})


def _edf_units(path: str) -> dict[str, str]:
    """Physical dimension of each channel, read from the EDF header.

    mne converts voltages to volts but leaves other units (``%``, ``mmHg``)
    as stored, and labels every EDF channel as a voltage, so the header is
    the only record of which channels were scaled.

    Args:
        path: Path to an EDF file.

    Returns:
        Channel label to physical dimension, both stripped.
    """
    with open(path, "rb") as f:
        header = f.read(256)
        n_signals = int(header[252:256].decode("ascii").strip())
        signal_header = f.read(256 * n_signals)
    field = 16 + 80  # label and transducer type come before the unit

    def column(offset: int, width: int) -> list[str]:
        start = offset * n_signals
        return [
            signal_header[start + i * width : start + (i + 1) * width]
            .decode("latin-1")
            .strip()
            for i in range(n_signals)
        ]

    return dict(zip(column(0, 16), column(field, 8)))


def _event_seconds(annotations: pd.DataFrame, names: frozenset[str]) -> set[int]:
    """Whole seconds covered by the annotations whose description is in names.

    An event covers the second ``floor(onset + k)`` for every 1 second chunk
    ``k`` that fits in its duration, so an event shorter than 1 second
    covers none. This is how the original code counts event seconds, with
    ``mne.events_from_annotations(..., chunk_duration=1.0)``.

    Args:
        annotations: Frame with ``onset``, ``duration`` and ``description``.
        names: Descriptions to keep.

    Returns:
        The covered seconds, counted from the start of the recording.
    """
    seconds: set[int] = set()
    selected = annotations[annotations["description"].isin(names)]
    for onset, duration in zip(selected["onset"], selected["duration"]):
        chunks = np.arange(0.0, float(duration), 1.0)
        chunks = chunks[float(duration) - chunks >= 1.0 - 1e-6]
        seconds.update(np.floor(float(onset) + chunks + 1e-9).astype(int).tolist())
    return seconds


def _resample(data: np.ndarray, sfreq: float, target: float, path: str) -> np.ndarray:
    """Resample the last axis of data from sfreq to target Hz.

    Args:
        data: Array whose last axis is time.
        sfreq: Rate of data in Hz.
        target: Output rate in Hz.
        path: Recording file, for the error message.

    Returns:
        The resampled array, or data itself when the rates are equal.

    Raises:
        ValueError: If target cannot be reached from sfreq with a ratio of
            integers up to 1000.
    """
    if math.isclose(sfreq, target):
        return data
    ratio = Fraction(target / sfreq).limit_denominator(_MAX_RESAMPLING_TERM)
    if ratio <= 0 or not math.isclose(sfreq * ratio, target, rel_tol=1e-9):
        raise ValueError(
            f"sampling_rate={target} Hz cannot be reached from the {sfreq} Hz of "
            f"{path} with a ratio of integers up to {_MAX_RESAMPLING_TERM}."
        )
    # padtype="line" continues the signal past its ends instead of padding
    # with zeros, which would distort SpO2 and CO2 at the edges.
    return sps.resample_poly(
        data, ratio.numerator, ratio.denominator, axis=-1, padtype="line"
    )


def detect_r_peaks(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """Find the R peaks of an ECG.

    The signal is band-pass filtered (3 Hz to 45 Hz, or less when the rate
    is low) and peaks at least 0.3 s apart and above 40% of the 99.5th
    percentile are kept. The original code uses biosppy's Hamilton
    segmenter; this simpler detector avoids the extra dependency.

    Args:
        ecg: One ECG channel.
        sfreq: Rate of ecg in Hz.

    Returns:
        Sample indices of the R peaks, in increasing order.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.sleep_apnea_detection import detect_r_peaks
        >>> ecg = np.zeros(1280)
        >>> ecg[64::128] = 1.0  # one beat per second at 128 Hz
        >>> detect_r_peaks(ecg, 128.0)[:3]
        array([ 64, 192, 320])
    """
    high = min(45.0, 0.45 * sfreq)
    numtaps = int(0.3 * sfreq) | 1
    if high <= 3.0 or len(ecg) <= 3 * numtaps:
        return np.array([], dtype=int)
    taps = sps.firwin(numtaps, [3.0, high], pass_zero=False, fs=sfreq)
    filtered = sps.filtfilt(taps, [1.0], ecg)
    height = 0.4 * np.percentile(filtered, 99.5)
    if height <= 0:
        return np.array([], dtype=int)
    peaks, _ = sps.find_peaks(
        filtered, distance=max(1, int(0.3 * sfreq)), height=height
    )
    return peaks


def ecg_features(
    ecg: np.ndarray, peaks: np.ndarray, sfreq: float, n_out: int, out_rate: float
) -> np.ndarray:
    """R-R interval and R-peak amplitude of one window, on the output grid.

    As in the original code, both are interpolated with a cubic spline over
    the window and are zero outside the span of the peaks. A window with 4
    peaks or fewer, or 200 or more, gives zeros.

    Args:
        ecg: ECG of the window at its recording rate.
        peaks: Sample indices of the R peaks in ecg.
        sfreq: Rate of ecg in Hz.
        n_out: Number of output samples.
        out_rate: Output rate in Hz.

    Returns:
        Array of shape ``(2, n_out)``: R-R interval in seconds, clipped to
        0-2 s, and R-peak amplitude in the unit of ecg.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.sleep_apnea_detection import ecg_features
        >>> ecg = np.zeros(1280)
        >>> peaks = np.arange(64, 1280, 96)  # one beat every 0.75 s at 128 Hz
        >>> ecg[peaks] = 1.0
        >>> out = ecg_features(ecg, peaks, 128.0, n_out=640, out_rate=64.0)
        >>> out.shape, round(float(out[0, 320]), 2)
        ((2, 640), 0.75)
    """
    out = np.zeros((2, n_out))
    if not 4 < len(peaks) < 200:
        return out
    grid = np.arange(n_out) / out_rate
    times = peaks / sfreq
    out[0] = splev(grid, splrep(times[1:], np.diff(times), k=3), ext=1)
    out[1] = splev(grid, splrep(times, ecg[peaks], k=3), ext=1)
    out[0] = np.clip(out[0], *_RRI_RANGE)
    return out


class SleepApneaDetection(BaseTask):
    """Sleep apnea detection on 30 second windows of a sleep study.

    Runs on :class:`~pyhealth.datasets.NCHSleepDataset` and
    :class:`~pyhealth.datasets.CHATDataset`. For each study this task:

    1. skips the study if its ``ahi`` is below ``min_ahi``
    2. reads the channels of the chosen ``modalities`` from the EDF (see
       below), re-referencing CHAT electrodes as the original code does
    3. resamples them to ``sampling_rate`` (64 Hz by default)
    4. cuts windows of ``window_seconds`` (30 by default) one after the
       other from the start of the recording, and drops windows that
       overlap a scored wake epoch
    5. labels each window 1 when an apnea or hypopnea covers at least one
       second of it, else 0

    Channels of each modality, NCH label or CHAT derivation:

    - ``eog``: ``EOG LOC-M2`` or ``E1 - M2``
    - ``eeg``: ``EEG C4-M1`` or ``C4 - M1``
    - ``resp``: ``Resp Airflow`` or ``Airflow``
    - ``spo2``: ``SpO2`` or ``SaO2``
    - ``co2``: ``Capno`` or ``Cap``
    - ``ecg``: ``ECG EKG2-EKG`` or ``ECG1 - ECG3``, turned into two channels,
      the R-R interval (seconds) and the R-peak amplitude, as in the
      original code

    Voltages are in microvolts; other channels keep the unit of the file
    (SpO2 in %, CO2 in mmHg). The event names counted as apnea, hypopnea
    and wake are those of
    :data:`~pyhealth.datasets.nch_sleep.NCH_APNEA_EVENTS`,
    :data:`~pyhealth.datasets.chat.CHAT_APNEA_EVENTS` and the matching
    hypopnea and stage constants.

    A study that lacks a channel of ``modalities`` is skipped with a
    warning, as in the original code. With ``fill_missing=True`` the
    missing modality is filled with zeros instead, which is how the
    original code represents a missing modality to its model. A study whose
    files cannot be read is skipped with a warning.

    The original code also balances the classes by dropping negative
    windows at random; this task keeps every window, so balance at training
    time if needed. Studies of one patient are related, so split by
    patient with :func:`~pyhealth.datasets.split_by_patient`.

    Each sample contains:

    - ``patient_id`` and ``record_id`` of the study
    - ``signal``: float32 array of shape
      ``(n_channels, window_seconds * sampling_rate)``, by default
      ``(7, 1920)``, channels in the order of :attr:`channel_names`
    - ``label``: 1 if apnea or hypopnea, else 0
    - ``apnea_seconds`` and ``hypopnea_seconds``: seconds of the window
      covered by each kind of event
    - ``start_second``: start of the window in the recording

    Args:
        modalities: Modalities to read, a subset of ``APNEA_MODALITIES``.
            The output channels always follow the order of
            ``APNEA_MODALITIES``.
        sampling_rate: Output rate in Hz.
        window_seconds: Window length in whole seconds.
        min_ahi: Studies whose ``ahi`` attribute is below this are skipped.
            The original code keeps NCH studies with an AHI of at least 3.
            None keeps every study.
        fill_missing: Fill a modality whose channels are missing with
            zeros instead of skipping the study.

    Raises:
        ValueError: If a modality is unknown or repeated, if
            ``modalities`` is empty, if ``sampling_rate`` is not a positive
            number, if ``window_seconds`` is not a positive integer, or, when
            a recording is read, if ``sampling_rate`` cannot be reached from
            its rate with a ratio of integers up to 1000.

    Examples:
        >>> from pyhealth.datasets import NCHSleepDataset, split_by_patient
        >>> from pyhealth.tasks import SleepApneaDetection
        >>> dataset = NCHSleepDataset(root="/path/to/nchsdb")  # doctest: +SKIP
        >>> task = SleepApneaDetection(min_ahi=3.0)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["signal"].shape  # doctest: +SKIP
        torch.Size([7, 1920])
        >>> SleepApneaDetection(modalities=["spo2", "ecg"]).channel_names
        ['spo2', 'ecg_rri', 'ecg_r_amplitude']
    """

    task_name: str = "SleepApneaDetection"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        modalities: Sequence[str] = APNEA_MODALITIES,
        sampling_rate: float = 64.0,
        window_seconds: int = 30,
        min_ahi: float | None = None,
        fill_missing: bool = False,
    ) -> None:
        super().__init__()
        modalities = list(modalities)
        unknown = [m for m in modalities if m not in APNEA_MODALITIES]
        if unknown or not modalities or len(set(modalities)) != len(modalities):
            raise ValueError(
                f"modalities must be distinct names from {APNEA_MODALITIES}, "
                f"got {modalities}."
            )
        if not (math.isfinite(sampling_rate) and sampling_rate > 0):
            raise ValueError(
                f"sampling_rate must be a positive number, got {sampling_rate}."
            )
        if isinstance(window_seconds, bool) or not (
            isinstance(window_seconds, int) and window_seconds > 0
        ):
            raise ValueError(
                f"window_seconds must be a positive integer, got {window_seconds}."
            )
        self.modalities = [m for m in APNEA_MODALITIES if m in modalities]
        self.sampling_rate = float(sampling_rate)
        self.window_seconds = window_seconds
        self.min_ahi = None if min_ahi is None else float(min_ahi)
        self.fill_missing = fill_missing

    @property
    def channel_names(self) -> list[str]:
        """Names of the output channels, in the order of ``signal``."""
        return [c for m in self.modalities for c in _OUTPUT_CHANNELS[m]]

    def read_modalities(
        self, signal_file: str
    ) -> tuple[dict[str, np.ndarray | None], float, int] | None:
        """Read the channels of each modality from an EDF file.

        Args:
            signal_file: Path to the EDF of the study.

        Returns:
            ``(signals, sfreq, n_samples)``: for each modality a 1-D array
            of ``n_samples`` at the recording rate ``sfreq``, in microvolts
            for voltages, or None when its channels are missing and
            ``fill_missing`` is set. None instead of the tuple when a
            channel is missing and ``fill_missing`` is not set.
        """
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        names = {name.upper(): name for name in raw.ch_names}
        chosen: dict[str, tuple[str, ...] | None] = {}
        for modality in self.modalities:
            chosen[modality] = next(
                (
                    tuple(names[c] for c in option)
                    for option in _MONTAGES[modality]
                    if all(c in names for c in option)
                ),
                None,
            )
        missing = [m for m, option in chosen.items() if option is None]
        if missing and not self.fill_missing:
            logger.warning(
                "Skipping %s, it has no channels for %s. Channels are %s.",
                signal_file,
                missing,
                raw.ch_names,
            )
            return None

        sfreq, n_samples = float(raw.info["sfreq"]), int(raw.n_times)
        picks = sorted({c for option in chosen.values() if option for c in option})
        if not picks:
            return {m: None for m in self.modalities}, sfreq, n_samples
        raw.pick(picks)
        raw.load_data(verbose="error")
        data = raw.get_data()
        units = _edf_units(signal_file)
        row = {name: i for i, name in enumerate(raw.ch_names)}
        for name, i in row.items():
            if units.get(name, "").lower() in _VOLTAGE_UNITS:
                data[i] *= 1e6
        signals: dict[str, np.ndarray | None] = {}
        for modality, option in chosen.items():
            if option is None:
                signals[modality] = None
            elif len(option) == 1:
                signals[modality] = data[row[option[0]]]
            else:
                signals[modality] = data[row[option[0]]] - data[row[option[1]]]
        return signals, sfreq, n_samples

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one study into window samples.

        Args:
            patient_id: Patient id of the study.
            event: A ``studies`` event with ``signal_file``,
                ``annotation_file``, ``record_id`` and ``ahi``.

        Returns:
            One sample dict per window kept.
        """
        record_id = str(getattr(event, "record_id", "") or "")
        if self.min_ahi is not None:
            try:
                ahi = float(getattr(event, "ahi", "nan"))
            except (TypeError, ValueError):
                ahi = math.nan
            if not ahi >= self.min_ahi:
                return []

        # Imported here: pyhealth.datasets imports pyhealth.tasks.
        from pyhealth.datasets.chat import (
            CHAT_APNEA_EVENTS,
            CHAT_HYPOPNEA_EVENTS,
            read_chat_annotations,
        )
        from pyhealth.datasets.nch_sleep import (
            NCH_APNEA_EVENTS,
            NCH_HYPOPNEA_EVENTS,
        )

        signal_file = str(getattr(event, "signal_file", "") or "")
        annotation_file = str(getattr(event, "annotation_file", "") or "")
        try:
            # Reads NSRR XML, or the onset/duration/description TSV of NCH.
            annotations = read_chat_annotations(annotation_file)
            loaded = self.read_modalities(signal_file)
        except (OSError, ValueError, RuntimeError) as err:
            logger.warning("Skipping %s, it cannot be read: %s", record_id, err)
            return []
        if loaded is None:
            return []
        signals, sfreq, n_native = loaded

        apnea = _event_seconds(annotations, NCH_APNEA_EVENTS | CHAT_APNEA_EVENTS)
        hypopnea = _event_seconds(
            annotations, NCH_HYPOPNEA_EVENTS | CHAT_HYPOPNEA_EVENTS
        )
        wake = _event_seconds(annotations, frozenset({"Sleep stage W", "Wake"}))

        n_windows = int(n_native / sfreq // self.window_seconds)
        win_native = round(self.window_seconds * sfreq)
        win_out = round(self.window_seconds * self.sampling_rate)
        if n_windows == 0 or win_out < 1:
            logger.warning(
                "Skipping %s, it is shorter than one %d s window.",
                record_id,
                self.window_seconds,
            )
            return []

        rows: list[np.ndarray] = []
        peaks = None
        for modality in self.modalities:
            series = signals[modality]
            if modality == "ecg":
                if series is not None:
                    peaks = detect_r_peaks(series, sfreq)
                continue
            if series is None:
                rows.append(np.zeros(n_windows * win_out))
            else:
                out = _resample(series, sfreq, self.sampling_rate, signal_file)
                out = np.pad(out, (0, max(0, n_windows * win_out - len(out))), "edge")
                rows.append(out[: n_windows * win_out])

        samples: list[dict[str, Any]] = []
        for w in range(n_windows):
            start = w * self.window_seconds
            window = set(range(start, start + self.window_seconds))
            if window & wake:
                continue
            channels = [r[w * win_out : (w + 1) * win_out] for r in rows]
            if "ecg" in self.modalities:
                ecg = np.zeros((2, win_out))
                if peaks is not None:
                    lo, hi = w * win_native, (w + 1) * win_native
                    inside = peaks[(peaks >= lo) & (peaks < hi)] - lo
                    ecg = ecg_features(
                        signals["ecg"][lo:hi],
                        inside,
                        sfreq,
                        win_out,
                        self.sampling_rate,
                    )
                channels.extend(ecg)
            apnea_seconds = len(window & apnea)
            hypopnea_seconds = len(window & hypopnea)
            samples.append(
                {
                    "patient_id": patient_id,
                    "record_id": record_id,
                    "start_second": start,
                    "signal": np.stack(channels).astype(np.float32),
                    "label": int(apnea_seconds + hypopnea_seconds > 0),
                    "apnea_seconds": apnea_seconds,
                    "hypopnea_seconds": hypopnea_seconds,
                }
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build window samples for one patient.

        Args:
            patient: A PyHealth patient whose ``studies`` events are rows of
                the NCH or CHAT metadata CSV.

        Returns:
            Window samples of all studies of the patient.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("studies"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples
